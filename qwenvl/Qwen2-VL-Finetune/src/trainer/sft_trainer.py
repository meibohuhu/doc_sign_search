import inspect
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from src.constants import IGNORE_INDEX

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenSFTTrainer, self).__init__(*args, **kwargs)
    
    ### mh: 2025-11-16: this is to ensure that the fg/bg pixels are not removed from the signature columns.
    def _set_signature_columns_if_needed(self):
        """
        Set the signature columns if they are not already set.
        For FBCF, we need to preserve pixel_values_videos_fg and pixel_values_videos_bg.
        We need to call the parent method first to get default columns, then add our custom ones.
        """
        # Call parent method to get default signature columns
        super()._set_signature_columns_if_needed()
        
        # Add custom FBCF columns that should not be removed
        if self._signature_columns is not None:
            # Ensure FBCF columns are included
            fbcf_columns = ["pixel_values_videos_fg", "pixel_values_videos_bg"]
            for col in fbcf_columns:
                if col not in self._signature_columns:
                    self._signature_columns.append(col)
################################################################################

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(QwenSFTTrainer, self)._save_checkpoint(model, trial)

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    #
    #     return super().training_step(model, inputs)

### mh 11/15/2025: FBCF loss functions 
    def _background_uniform_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between model output and uniform distribution.
        Memory-optimized version: computes KL efficiently with minimal intermediate storage.
        
        KL(p||u) = sum_i p_i * log(p_i / u_i) = sum_i p_i * log(p_i) - sum_i p_i * log(u_i)
                 = sum_i p_i * log(p_i) - log(1/vocab_size) * sum_i p_i
                 = sum_i p_i * log(p_i) + log(vocab_size)
        
        Note: We still need to compute probs for KL, but we sum immediately to reduce memory.
        """
        vocab_size = logits.size(-1)
        # Use bool mask for better memory efficiency
        label_mask = (labels != IGNORE_INDEX).to(logits.device)
        mask_sum = label_mask.sum()
        if mask_sum == 0:
            return logits.new_zeros(())
        
        uniform_log_prob = math.log(vocab_size)
        
        # Compute log_softmax (more stable than softmax then log)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute KL divergence per token: sum(exp(log_probs) * log_probs) + uniform_log_prob
        # We compute probs but immediately reduce dimension by summing over vocab
        # This reduces memory compared to keeping full probs tensor around
        probs = log_probs.exp()  # [batch_size, seq_len, vocab_size]
        # Sum over vocab dimension immediately: [batch_size, seq_len]
        kl_per_token = (probs * log_probs).sum(dim=-1) + uniform_log_prob
        
        # Apply mask and compute mean (avoid recomputing mask_sum)
        kl = kl_per_token * label_mask.to(kl_per_token.dtype)
        kl_sum = kl.sum()
        # Convert mask_sum to same dtype as kl_per_token for division
        mask_sum_float = mask_sum.to(kl_per_token.dtype)
        
        # Note: del doesn't immediately free GPU memory, but helps with reference counting
        # The tensors will be freed when out of scope anyway
        del log_probs, probs, kl_per_token, kl
        
        return kl_sum / mask_sum

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        if not getattr(self.args, "enable_fbcf", False):
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        # Check if using single-path sampling mode
        use_sampling = getattr(self.args, "fbcf_sampling_mode", False)
        
        shared_inputs = inputs.copy()
        fg_pixels = shared_inputs.pop("pixel_values_videos_fg", None)
        bg_pixels = shared_inputs.pop("pixel_values_videos_bg", None)

        # Debug: Check if fg_pixels and bg_pixels are correctly applied
        if "pixel_values_videos" in shared_inputs:
            original_pixels = shared_inputs["pixel_values_videos"]
            
            # Only check once and warn if fg/bg pixels are missing
            if not hasattr(self, '_fbcf_warn_checked'):
                self._fbcf_warn_checked = True
                
                if fg_pixels is None or bg_pixels is None:
                    # Check if mask_folder is set (from train_dataset.data_args)
                    mask_folder = None
                    use_masks = False
                    if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                        if hasattr(self.train_dataset, 'data_args'):
                            mask_folder = getattr(self.train_dataset.data_args, 'mask_folder', None)
                        if hasattr(self.train_dataset, 'use_masks'):
                            use_masks = self.train_dataset.use_masks
                    
                    if mask_folder is None:
                        print(f"⚠️  CRITICAL: mask_folder is not set! FBCF requires --mask_folder parameter.")
                        print(f"   Add --mask_folder <path> to your training script to enable FBCF masks.")
                    elif not use_masks:
                        print(f"⚠️  mask_folder is set ({mask_folder}), but use_masks is False.")
                        print(f"   This might indicate the mask_folder path doesn't exist or is invalid.")
                    else:
                        print(f"⚠️  mask_folder is set ({mask_folder}), use_masks is True, but fg/bg_pixels are None.")
                        print(f"   Check if mask files exist in the mask_folder directory or if mask loading failed.")

        if use_sampling:
            # Single-path sampling: randomly select one view per sample
            # Configurable ratios: default 50% original, 35% foreground, 15% background
            if "pixel_values_videos" not in shared_inputs:
                # Fallback to standard loss if no video input
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
            
            # Note: In sampling mode, pixel_values_videos shape is [num_patches, feature_dim] (flattened)
            # video_grid_thw shape is [num_videos_in_batch, 3] where each row is [T, H, W]
            # For single-path sampling, we need to select samples at the sample level, not video level
            # Since DataCollator concatenates all videos, we need to determine the actual batch size
            # from other inputs like input_ids or labels
            if "input_ids" in shared_inputs:
                batch_size = shared_inputs["input_ids"].shape[0]
            elif "labels" in shared_inputs:
                batch_size = shared_inputs["labels"].shape[0]
            elif "video_grid_thw" in shared_inputs:
                # Fallback: assume each sample has one video (common case)
                video_grid_thw = shared_inputs["video_grid_thw"]
                batch_size = video_grid_thw.shape[0]
            else:
                # Last resort: default to 1
                batch_size = 1
            
            device = shared_inputs["pixel_values_videos"].device if isinstance(shared_inputs["pixel_values_videos"], torch.Tensor) else torch.device("cuda")
            
            # IMPORTANT: In distributed training (DeepSpeed/DDP), we need to ensure all processes
            # select the same view type per step to avoid gradient sync issues.
            # Solution: Use step-based deterministic random selection. Each step, all processes
            # select the same view type (original, foreground, or background) for all samples.
            # This ensures:
            # 1. All processes in the same step use the same view type (no sync issues)
            # 2. Different steps select different view types randomly based on ratios
            # 3. Over time, the distribution matches the configured ratios
            
            # Get sampling ratios (with normalization to ensure they sum to 1.0)
            ratio_orig = getattr(self.args, "fbcf_sampling_ratio_original", 0.4)
            ratio_fg = getattr(self.args, "fbcf_sampling_ratio_foreground", 0.4)
            ratio_bg = getattr(self.args, "fbcf_sampling_ratio_background", 0.2)
            
            # Normalize ratios to sum to 1.0
            total_ratio = ratio_orig + ratio_fg + ratio_bg
            if total_ratio > 0:
                ratio_orig = ratio_orig / total_ratio
                ratio_fg = ratio_fg / total_ratio
                ratio_bg = ratio_bg / total_ratio
            else:
                # Fallback to default if all ratios are 0
                ratio_orig, ratio_fg, ratio_bg = 0.4, 0.4, 0.2
            
            # Get current training step for deterministic seed
            step = getattr(self.state, 'global_step', 0) if hasattr(self, 'state') else 0
            
            # Generate a single deterministic random value for this step
            # This ensures all processes select the same view type for this step
            import random
            random.seed(step)
            rand_value = random.random()  # [0, 1)
            
            # Determine view type for this step based on ratios
            # All samples in this step will use the same view type
            threshold_orig = ratio_orig
            threshold_fg = ratio_orig + ratio_fg
            if rand_value < threshold_orig:
                view_type = 0  # original
                view_name = "original"
            elif rand_value < threshold_fg:
                view_type = 1  # foreground
                view_name = "foreground"
            else:
                view_type = 2  # background
                view_name = "background"
            
            # All samples use the same view type in this step
            use_full = torch.tensor(view_type == 0, device=device).expand(batch_size)
            use_fg = torch.tensor(view_type == 1, device=device).expand(batch_size)
            use_bg = torch.tensor(view_type == 2, device=device).expand(batch_size)
            
            # Debug: Log sampling distribution
            print(f"📊 Sampling (step {step}): {view_name} for all {batch_size} samples (rand={rand_value:.3f})", flush=True)
            
            total_loss = None
            log_metrics = {}
            
            # Since all samples use the same view type in this step, we can simplify the forward pass
            # Helper function to prepare inputs for all samples with a specific pixel source
            # Since all samples use the same view type in this step, forward all samples together
            if view_type == 0:
                # Original view: standard CE loss
                model_inputs = {k: v for k, v in shared_inputs.items() 
                              if k not in ["pixel_values_videos_fg", "pixel_values_videos_bg"]}
                outputs = model(**model_inputs)
                total_loss = outputs.loss
                log_metrics = {"loss_full": total_loss.detach()}
                
            elif view_type == 1:
                # Foreground view: CE loss with fg_loss_weight
                if fg_pixels is None:
                    # Fallback to original if fg_pixels not available
                    print(f"⚠️  WARNING: fg_pixels is None, falling back to original view", flush=True)
                    model_inputs = {k: v for k, v in shared_inputs.items() 
                                  if k not in ["pixel_values_videos_fg", "pixel_values_videos_bg"]}
                    outputs = model(**model_inputs)
                    total_loss = outputs.loss
                    log_metrics = {"loss_full": total_loss.detach()}
                else:
                    # Use foreground pixels
                    model_inputs = {k: v for k, v in shared_inputs.items() 
                                  if k not in ["pixel_values_videos", "pixel_values_videos_bg"]}
                    model_inputs["pixel_values_videos"] = fg_pixels
                    outputs = model(**model_inputs)
                    total_loss = self.args.fg_loss_weight * outputs.loss
                    log_metrics = {"loss_fg": outputs.loss.detach()}
                    
            else:  # view_type == 2
                # Background view: KL loss for uniform distribution
                if bg_pixels is None:
                    # Fallback to original if bg_pixels not available
                    print(f"⚠️  WARNING: bg_pixels is None, falling back to original view", flush=True)
                    model_inputs = {k: v for k, v in shared_inputs.items() 
                                  if k not in ["pixel_values_videos_fg", "pixel_values_videos_bg"]}
                    outputs = model(**model_inputs)
                    total_loss = outputs.loss
                    log_metrics = {"loss_full": total_loss.detach()}
                else:
                    # Use background pixels
                    model_inputs = {k: v for k, v in shared_inputs.items() 
                                  if k not in ["pixel_values_videos", "pixel_values_videos_fg"]}
                    model_inputs["pixel_values_videos"] = bg_pixels
                    labels = model_inputs.get("labels")
                    if labels is None:
                        raise ValueError("Labels are required to compute background KL loss.")
                    
                    # For background KL loss, we need logits, not just loss
                    # In DeepSpeed ZeRO-3, when labels are provided, the model may only return loss
                    # So we need to remove labels temporarily to get logits, then compute KL loss manually
                    # Memory optimization: disable unnecessary outputs to reduce memory usage
                    model_inputs_for_logits = {k: v for k, v in model_inputs.items() if k != "labels"}
                    model_inputs_for_logits["output_attentions"] = False
                    model_inputs_for_logits["output_hidden_states"] = False
                    
                    # Forward pass to get logits
                    # Note: Keep autocast enabled to match training precision (bf16/fp16)
                    outputs = model(**model_inputs_for_logits)
                    
                    # Get logits from outputs
                    bg_logits = getattr(outputs, "logits", None)
                    if bg_logits is None:
                        raise ValueError("Cannot get logits from model outputs for background KL loss computation. "
                                       "The model must return logits when labels are not provided.")
                    
                    # Compute background KL loss (memory-optimized)
                    loss_bg = self._background_uniform_loss(bg_logits, labels)
                    
                    # Immediately free logits to reduce memory pressure
                    # Note: In DeepSpeed training, memory is managed automatically, so manual
                    # cache clearing may not be necessary and can cause performance overhead
                    del bg_logits, outputs
                    # Only clear cache if not using DeepSpeed (to avoid performance penalty)
                    # In DeepSpeed, the framework manages memory more efficiently
                    use_deepspeed = hasattr(self.args, 'deepspeed') and self.args.deepspeed is not None
                    if torch.cuda.is_available() and not use_deepspeed:
                        torch.cuda.empty_cache()  # Clear GPU cache (only for non-DeepSpeed)
                    
                    total_loss = self.args.fbcf_lambda * self.args.bg_loss_weight * loss_bg
                    log_metrics = {"loss_bg": loss_bg.detach()}
            
        
            # Log metrics
            if log_metrics:
                scalar_logs = {}
                for key, value in log_metrics.items():
                    if torch.is_tensor(value):
                        scalar_logs[key] = value.detach().float().mean().item()
                    else:
                        scalar_logs[key] = value
                self.log(scalar_logs)
            
            # Return loss and outputs if requested
            # Note: In step-based sampling mode, we forward all samples with the same view type,
            # so we can directly return the outputs from this step's forward pass.
            if return_outputs:
                return (total_loss, outputs)
            else:
                return total_loss
        else:
            # Full forward mode (original implementation)
            outputs = model(**shared_inputs)
            total_loss = outputs.loss   ## # 原始video的标准CE loss
            log_metrics = {"loss_full": total_loss.detach()}

            if fg_pixels is not None:
                fg_inputs = dict(shared_inputs)
                fg_inputs["pixel_values_videos"] = fg_pixels
                fg_outputs = model(**fg_inputs)
                loss_fg = fg_outputs.loss
                total_loss = total_loss + self.args.fg_loss_weight * loss_fg   # 前景video的CE loss  
                log_metrics["loss_fg"] = loss_fg.detach()

            if bg_pixels is not None:
                bg_inputs = dict(shared_inputs)
                labels = bg_inputs.pop("labels", None)
                if labels is None:
                    raise ValueError("Labels are required to compute background KL loss.")
                bg_inputs["pixel_values_videos"] = bg_pixels
                bg_outputs = model(**bg_inputs)
                loss_bg = self._background_uniform_loss(bg_outputs.logits, labels)
                total_loss = total_loss + self.args.fbcf_lambda * self.args.bg_loss_weight * loss_bg # 背景video的KL散度（鼓励均匀分布）
                log_metrics["loss_bg"] = loss_bg.detach()

            if log_metrics:
                scalar_logs = {}
                for key, value in log_metrics.items():
                    if torch.is_tensor(value):
                        scalar_logs[key] = value.detach().float().mean().item()
                    else:
                        scalar_logs[key] = value
                self.log(scalar_logs)
            
            # print(f"log_metrics: {log_metrics}")
            print(f"total_loss: {total_loss}")
            return (total_loss, outputs) if return_outputs else total_loss
################################################################################

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels") if "labels" in inputs else None

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            logits = outputs.logits if hasattr(outputs, "logits") else None

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)
