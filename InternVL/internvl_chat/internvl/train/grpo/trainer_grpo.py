"""
InternVL GRPO Trainer — adapted for InternVL2.5 architecture.
Handles vision-language GRPO with pixel_values + image_flags.

Simplified approach:
- Each GPU gets per_device_train_batch_size unique samples
- num_generations completions are generated per sample via a loop
- Rewards and advantages computed locally (no cross-GPU gather needed for rewards)
"""

import os
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import transformers
from accelerate.utils import set_seed
from packaging import version
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.trainer import TRAINER_STATE_NAME, PREFIX_CHECKPOINT_DIR

from trl.trainer.utils import selective_log_softmax
from trl.trainer.grpo_config import GRPOConfig
from trl.models import unwrap_model_for_generation

from internvl.train.constants import IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN
from internvl.conversation import get_conv_template

RewardFunc = Union[PreTrainedModel, Callable]


# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_peft_model(model) -> bool:
    try:
        from peft import PeftModel
        return isinstance(model, PeftModel)
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
#  GRPO Training Arguments (extends TRL GRPOConfig)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GRPOTrainingArguments(GRPOConfig):
    """Extends TRL GRPOConfig with extra convenience args."""
    reward_weights_str: Optional[str] = field(
        default=None,
        metadata={'help': 'Comma-separated reward weights, e.g. "0.5,0.5"'}
    )
    log_completions: bool = field(default=True)
    num_completions_to_print: int = field(default=4)


# ──────────────────────────────────────────────────────────────────────────────
#  Data collator (top-level for pickling)
# ──────────────────────────────────────────────────────────────────────────────

def _grpo_data_collator(features):
    """Identity collator — returns list of dicts as-is."""
    return features


# ──────────────────────────────────────────────────────────────────────────────
#  InternVLGRPOTrainer
# ──────────────────────────────────────────────────────────────────────────────

class InternVLGRPOTrainer(Trainer):
    """
    GRPO Trainer for InternVL2.5 vision-language models.

    Simplified design:
    - Each GPU processes per_device_train_batch_size unique prompts
    - For each prompt, num_generations completions are generated sequentially
    - Rewards/advantages computed locally per GPU (each GPU has complete groups)
    - No cross-GPU gather needed for reward normalization
    """

    def __init__(
        self,
        model: PreTrainedModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOTrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers=(None, None),
        # InternVL-specific
        num_image_token: int = 64,
        conv_style: str = 'internvl2_5',
    ):
        # ── Reference model ──
        self.beta = args.beta
        self.ref_model = None  # With LoRA, use disable_adapter() for reference

        # ── Processing class (tokenizer) ──
        if processing_class is None:
            raise ValueError("processing_class (tokenizer) must be provided")
        self.tokenizer = processing_class

        # ── InternVL-specific ──
        self.num_image_token = num_image_token
        self.conv_style = conv_style
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        # ── Reward functions ──
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if isinstance(reward_func, nn.Module):
                self.reward_func_names.append(reward_func.config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_func.__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # ── GRPO hyperparameters ──
        self.max_prompt_length = getattr(args, 'max_prompt_length', 4096)
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = getattr(args, 'mask_truncated_completions', False)

        # EOS token from conv template
        template = get_conv_template(self.conv_style)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or 0,
            eos_token_id=eos_token_id,
            temperature=self.temperature,
            top_p=getattr(args, 'top_p', 1.0),
            top_k=getattr(args, 'top_k', 50),
        )

        # ── Multi-step ──
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        # ── Initialize parent Trainer ──
        super().__init__(
            model=model,
            args=args,
            data_collator=_grpo_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # ── Metrics ──
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print

        # ── Validate ──
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least 2 generations per prompt")

        set_seed(args.seed, device_specific=True)
        self.model_accepts_loss_kwargs = False

    # ──────────────────────────────────────────────────────────────────────────
    #  Dataloader — use default Trainer sampler (DistributedSampler for DDP)
    # ──────────────────────────────────────────────────────────────────────────

    def _set_signature_columns_if_needed(self):
        # Include all dataset fields so _remove_unused_columns doesn't strip them
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_text", "ground_truth", "pixel_values",
                "image_flags", "num_patches",
            ]

    # ──────────────────────────────────────────────────────────────────────────
    #  InternVL-specific: Prompt formatting
    # ──────────────────────────────────────────────────────────────────────────

    def _format_prompt(self, prompt_text: str, num_patches: int) -> str:
        """Format a prompt with InternVL image tokens for video frames.
        Must match SFT training format: Frame-1: <img>...</img>\nFrame-2: <img>...</img>\n...
        """
        frame_tokens = []
        for i in range(num_patches):
            img_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
            frame_tokens.append(f'Frame-{i + 1}: {img_tokens}')
        video_placeholder = '\n'.join(frame_tokens)

        template = get_conv_template(self.conv_style)
        template.system_message = getattr(
            self.accelerator.unwrap_model(self.model), 'system_message', template.system_message
        )
        question = video_placeholder + '\n' + prompt_text
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    # ──────────────────────────────────────────────────────────────────────────
    #  Per-token log probabilities
    # ──────────────────────────────────────────────────────────────────────────

    def _get_per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_flags: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Compute per-token log probabilities using InternVL forward pass."""
        unwrapped = self.accelerator.unwrap_model(model)
        unwrapped.img_context_token_id = self.img_context_token_id

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]  # (B, L-1, V)
        input_ids_shifted = input_ids[:, 1:]

        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
            input_ids_shifted = input_ids_shifted[:, -logits_to_keep:]

        logits = logits / self.temperature
        logps = selective_log_softmax(logits, input_ids_shifted)
        return logps

    # ──────────────────────────────────────────────────────────────────────────
    #  Generate & score completions
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_inputs(self, inputs):
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict],
    ) -> dict[str, Any]:
        """
        For each input sample, generate num_generations completions, compute rewards,
        and return expanded tensors (batch_size * num_generations).

        This is done locally per GPU — no cross-GPU gather needed since each GPU
        has complete groups for advantage normalization.
        """
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"
        G = self.num_generations
        B = len(inputs)  # per-device batch size (unique prompts)

        # ── Step 1: Prepare prompts ──
        prompt_texts = [x["prompt_text"] for x in inputs]
        ground_truths = [x["ground_truth"] for x in inputs]
        all_pixel_values = [x["pixel_values"] for x in inputs]
        all_image_flags = [x["image_flags"] for x in inputs]
        all_num_patches = [x["num_patches"] for x in inputs]

        # Format prompts with image tokens
        formatted_prompts = []
        for pt, np_ in zip(prompt_texts, all_num_patches):
            formatted_prompts.append(self._format_prompt(pt, np_))

        # Tokenize with left padding for generation
        self.tokenizer.padding_side = 'left'
        tokenized = self.tokenizer(
            formatted_prompts,
            return_tensors='pt',
            padding=True,
            truncation=False,
        )
        prompt_ids = tokenized['input_ids'].to(device)       # (B, P)
        prompt_mask = tokenized['attention_mask'].to(device)  # (B, P)

        # Concatenate pixel_values and image_flags across batch
        pixel_values = torch.cat(all_pixel_values, dim=0).to(device=device, dtype=torch.bfloat16)  # (total_frames, 3, H, W)
        image_flags = torch.cat(all_image_flags, dim=0).to(device=device)  # (total_frames,)

        # ── Step 2: Generate G completions per prompt ──
        all_completion_ids = []
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator,
            gather_deepspeed3_params=getattr(self.args, 'ds3_gather_for_generation', False),
        ) as unwrapped_model:
            unwrapped_model.img_context_token_id = self.img_context_token_id
            for g in range(G):
                with torch.no_grad():
                    gen_output = unwrapped_model.generate(
                        pixel_values=pixel_values,
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        generation_config=self.generation_config,
                    )
                # Extract completion tokens
                prompt_length = prompt_ids.size(1)
                if gen_output.size(1) > prompt_length:
                    comp_ids = gen_output[:, prompt_length:]  # (B, C_g)
                else:
                    comp_ids = gen_output
                all_completion_ids.append(comp_ids)

        # Pad completions to same length and interleave: [p0g0, p0g1, p1g0, p1g1, ...]
        max_comp_len = max(c.size(1) for c in all_completion_ids)
        padded_completions = []
        for comp_ids in all_completion_ids:
            if comp_ids.size(1) < max_comp_len:
                pad_len = max_comp_len - comp_ids.size(1)
                comp_ids = F.pad(comp_ids, (0, pad_len), value=self.tokenizer.pad_token_id or 0)
            padded_completions.append(comp_ids)

        # Stack: (G, B, C) -> interleave to (B*G, C) with grouping [p0g0, p0g1, p1g0, p1g1, ...]
        stacked = torch.stack(padded_completions, dim=1)  # (B, G, C)
        completion_ids = stacked.reshape(B * G, max_comp_len)  # (B*G, C)

        # Expand prompt_ids/mask: repeat each prompt G times
        prompt_ids_exp = prompt_ids.repeat_interleave(G, dim=0)    # (B*G, P)
        prompt_mask_exp = prompt_mask.repeat_interleave(G, dim=0)  # (B*G, P)

        # Expand pixel_values and image_flags: each sample's frames repeated G times
        # pixel_values shape: (total_frames, 3, H, W) where total_frames = sum(num_patches)
        # We need to repeat each sample's frames G times
        expanded_pv_list = []
        expanded_if_list = []
        offset = 0
        for np_ in all_num_patches:
            sample_pv = pixel_values[offset:offset + np_]  # (np_, 3, H, W)
            sample_if = image_flags[offset:offset + np_]   # (np_,)
            for _ in range(G):
                expanded_pv_list.append(sample_pv)
                expanded_if_list.append(sample_if)
            offset += np_
        pixel_values_exp = torch.cat(expanded_pv_list, dim=0)  # (total_frames * G, 3, H, W)
        image_flags_exp = torch.cat(expanded_if_list, dim=0)

        # Expand ground_truths: repeat each G times
        ground_truths_exp = []
        for gt in ground_truths:
            ground_truths_exp.extend([gt] * G)

        # ── Step 3: Build completion mask ──
        full_ids = torch.cat([prompt_ids_exp, completion_ids], dim=1)
        eos_token_id = self.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask_exp, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # ── Step 4: Compute log probabilities ──
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, full_ids, attention_mask,
                    pixel_values_exp, image_flags_exp, logits_to_keep,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif is_peft_model(self.model):
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, full_ids, attention_mask,
                        pixel_values_exp, image_flags_exp, logits_to_keep,
                    )
            else:
                ref_per_token_logps = None

        # ── Step 5: Decode and compute rewards ──
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        template = get_conv_template(self.conv_style)
        sep = template.sep.strip() if template.sep else ""
        if sep:
            completions_text = [c.split(sep)[0].strip() for c in completions_text]

        # Compute rewards locally (no gather needed — each GPU has complete groups)
        num_completions = B * G
        rewards_per_func = torch.zeros(num_completions, len(self.reward_funcs), device=device)
        for i, (reward_func, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_func_names)
        ):
            if isinstance(reward_func, nn.Module):
                pass  # not used for rule-based
            else:
                output_rewards = reward_func(
                    completions=completions_text,
                    ground_truths=ground_truths_exp,
                )
                output_rewards = [r if r is not None else torch.nan for r in output_rewards]
                rewards_per_func[:, i] = torch.tensor(output_rewards, dtype=torch.float32, device=device)

        # ── Step 6: Compute advantages locally ──
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)  # (B*G,)

        # Group rewards by prompt: (B, G)
        rewards_grouped = rewards.view(B, G)
        mean_grouped = rewards_grouped.mean(dim=1, keepdim=True)  # (B, 1)
        std_grouped = rewards_grouped.std(dim=1, keepdim=True)    # (B, 1)

        # Expand back to (B*G,)
        mean_expanded = mean_grouped.repeat(1, G).view(-1)
        std_expanded = std_grouped.repeat(1, G).view(-1)

        advantages = rewards - mean_expanded
        if self.scale_rewards and self.scale_rewards not in (False, 'false', 'no', 'none'):
            advantages = advantages / (std_expanded + 1e-6)###### 本来写的是1e-4, follow verl-agent

        # ── Step 7: Log metrics ──
        if mode == "train":
            if not hasattr(self.state, 'num_input_tokens_seen'):
                self.state.num_input_tokens_seen = 0
            self.state.num_input_tokens_seen += attention_mask.sum().item()
        self._metrics[mode]["num_tokens"] = [getattr(self.state, 'num_input_tokens_seen', 0)]
        self._metrics[mode]["completions/mean_length"].append(completion_mask.sum(1).float().mean().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_expanded.mean().item())
        self._metrics[mode]["reward_std"].append(std_expanded.mean().item())

        # Print sample completions
        if self.accelerator.is_main_process and self.log_completions:
            n = min(self.num_completions_to_print, num_completions)
            for j in range(n):
                print(f"\n--- Sample {j} (prompt {j // G}, gen {j % G}) ---")
                print(f"  GT: {ground_truths_exp[j][:100]}...")
                print(f"  Gen: {completions_text[j][:100]}...")
                r_strs = [f"{self.reward_func_names[k]}={rewards_per_func[j, k]:.3f}" for k in range(len(self.reward_funcs))]
                print(f"  Rewards: {', '.join(r_strs)}, Advantage: {advantages[j]:.3f}")

        return {
            "prompt_ids": prompt_ids_exp,
            "prompt_mask": prompt_mask_exp,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "pixel_values": pixel_values_exp,
            "image_flags": image_flags_exp,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Loss computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("InternVLGRPOTrainer does not support return_outputs")

        # inputs is a list of dicts from data_collator on first call
        if isinstance(inputs, list):
            inputs = self._generate_and_score_completions(inputs)

        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values = inputs["pixel_values"]
        image_flags = inputs["image_flags"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask,
            pixel_values, image_flags, logits_to_keep,
        )

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # GRPO clipped surrogate loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).nanmean().item()
        )

        return loss

    # ──────────────────────────────────────────────────────────────────────────
    #  Logging
    # ──────────────────────────────────────────────────────────────────────────

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if val}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()

    # ──────────────────────────────────────────────────────────────────────────
    #  Checkpoint saving
    # ──────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, model, trial):
        """Save checkpoint with LoRA adapter support."""
        if is_peft_model(self.model):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            if self.hp_search_backend is None and trial is None:
                self.store_flos()
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(output_dir)
                self._save_rng_state(output_dir)

            if self.args.should_save:
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
        else:
            super()._save_checkpoint(model, trial)
