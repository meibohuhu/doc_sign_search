"""
InternVL GRPO Training Script for Sign Language Translation.
Loads SFT checkpoint, applies LoRA, and trains with GRPO using BLEU + BERTScore rewards.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)

from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from internvl.train.grpo.grpo_dataset import GRPOVideoDataset
from internvl.train.grpo.reward_functions import bertscore_reward, bleu_reward, bleu1_reward, rouge_reward
from internvl.train.grpo.trainer_grpo import InternVLGRPOTrainer, GRPOTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    freeze_llm: bool = field(default=True)
    freeze_backbone: bool = field(default=False)
    freeze_mlp: bool = field(default=True)
    use_backbone_lora: int = field(default=0)
    use_llm_lora: int = field(default=16)
    vision_select_layer: int = field(default=-1)
    grad_checkpoint: bool = field(default=True)
    drop_path_rate: float = field(default=0.0)
    ps_version: Literal['v1', 'v2'] = field(default='v2')
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={'help': 'Path to JSONL data file'})
    video_root: str = field(default=None, metadata={'help': 'Root directory for video files'})
    force_image_size: int = field(default=224)
    down_sample_ratio: float = field(default=0.5)
    conv_style: str = field(default='internvl2_5')
    min_num_frame: int = field(default=8)
    max_num_frame: int = field(default=64)
    sampling_method: str = field(default='fps16.0')
    normalize_type: str = field(default='imagenet')


def main():
    os.environ['ACCELERATE_GRADIENT_ACCUMULATION_STEPS'] = '1'

    # Init distributed
    launcher = os.environ.get('LAUNCHER', 'slurm')
    dist_backend = os.environ.get('DIST_BACKEND', 'nccl')
    init_dist(launcher=launcher, backend=dist_backend)

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Parse reward weights
    if training_args.reward_weights_str:
        training_args.reward_weights = [float(w) for w in training_args.reward_weights_str.split(',')]

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, '
        f'n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}'
    )

    set_seed(training_args.seed)

    # ── Load tokenizer ──
    tokenizer_path = model_args.model_name_or_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.model_max_length = 8192

    token_list = [
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
        REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # ── Load model ──
    logger.info(f'Loading InternVLChatModel from {model_args.model_name_or_path}...')
    config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    if config.llm_config.model_type == 'internlm2':
        config.llm_config.attn_implementation = 'flash_attention_2'
    else:
        config.llm_config._attn_implementation = 'flash_attention_2'
    config.template = data_args.conv_style
    config.select_layer = model_args.vision_select_layer
    config.ps_version = model_args.ps_version
    config.dynamic_image_size = False
    config.use_thumbnail = False
    if config.downsample_ratio != data_args.down_sample_ratio:
        config.downsample_ratio = data_args.down_sample_ratio

    model = InternVLChatModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        config=config,
    )
    model.img_context_token_id = img_context_token_id

    # Handle image size
    patch_size = model.config.vision_config.patch_size
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from {model.config.vision_config.image_size} to {data_args.force_image_size}')
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2)
    )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False

    # ── Freeze / unfreeze ──
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    # ── Apply LoRA ──
    # Check if model already has LoRA (e.g., loaded from SFT checkpoint with LoRA)
    def _has_peft(module):
        try:
            from peft import PeftModel
            return isinstance(module, PeftModel)
        except ImportError:
            return False

    if model_args.use_backbone_lora:
        if not _has_peft(model.vision_model):
            model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
            logger.info('Applied new backbone LoRA')
        else:
            logger.info('Backbone already has LoRA from checkpoint — skipping wrap_backbone_lora')
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        if not _has_peft(model.language_model):
            model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
            logger.info('Applied new LLM LoRA')
        else:
            logger.info('LLM already has LoRA from checkpoint — skipping wrap_llm_lora')
            # Ensure input_require_grads is enabled (needed for LoRA training)
            model.language_model.enable_input_require_grads()
        model.config.use_llm_lora = model_args.use_llm_lora

    # Ensure LoRA params are trainable (freeze_llm may have frozen them)
    if model_args.use_llm_lora and model_args.freeze_llm:
        for name, param in model.language_model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

    # ── Gradient checkpointing ──
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        if hasattr(model.language_model, '_set_gradient_checkpointing'):
            model.language_model._set_gradient_checkpointing()
        elif hasattr(model.language_model, 'base_model') and hasattr(model.language_model.base_model, '_set_gradient_checkpointing'):
            model.language_model.base_model._set_gradient_checkpointing()
        else:
            # PeftModel: enable gradient checkpointing on the underlying model
            model.language_model.gradient_checkpointing_enable()

    # Print trainable parameters
    if dist.get_rank() == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f'Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)')
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f'  trainable: {name}')

    # ── Build dataset ──
    logger.info(f'Building GRPO dataset from {data_args.data_path}...')
    train_dataset = GRPOVideoDataset(
        jsonl_path=data_args.data_path,
        video_root=data_args.video_root,
        image_size=data_args.force_image_size,
        max_num_frame=data_args.max_num_frame,
        min_num_frame=data_args.min_num_frame,
        sampling_method=data_args.sampling_method,
        normalize_type=data_args.normalize_type,
    )

    # ── Build reward functions ──
    reward_funcs = [bleu1_reward, rouge_reward]  # weights: 0.5,0.5
    logger.info(f'Reward functions: {[f.__name__ for f in reward_funcs]}')
    logger.info(f'Reward weights: {training_args.reward_weights}')

    # ── Create trainer ──
    num_image_token = model.num_image_token
    logger.info(f'num_image_token: {num_image_token}')

    trainer = InternVLGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        num_image_token=num_image_token,
        conv_style=data_args.conv_style,
    )

    # ── Train ──
    logger.info('Starting GRPO training...')
    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    logger.info('GRPO training complete!')


if __name__ == '__main__':
    main()
