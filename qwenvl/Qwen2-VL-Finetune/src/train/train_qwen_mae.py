#!/usr/bin/env python3
"""
MAE pre-training script for Qwen2-VL vision encoder
Supports DeepSpeed for efficient multi-GPU training
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import transformers
from transformers import AutoProcessor, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import TrainerCallback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.qwen_mae import QwenViTMAE
from src.dataset.mae_dataset import make_mae_data_module


class MemoryClearingCallback(TrainerCallback):
    """Callback to clear GPU cache after each training step to reduce memory pressure"""
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Clear GPU cache after each training step"""
        try:
            # Use DeepSpeed's accelerator for multi-GPU compatibility
            if args.deepspeed is not None:
                try:
                    from deepspeed import get_accelerator
                    get_accelerator().empty_cache()
                except ImportError:
                    # Fallback to torch.cuda if DeepSpeed accelerator is not available
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # For non-DeepSpeed training, use torch.cuda
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            # Ignore any errors during cache clearing
            pass


class MAETrainer(Trainer):
    """Custom Trainer for MAE pre-training"""
    
    def __init__(self, mask_ratio=0.75, **kwargs):
        """Initialize MAETrainer with mask_ratio"""
        self.mask_ratio = mask_ratio
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute MAE reconstruction loss
        
        Args:
            model: QwenViTMAE model
            inputs: dict with 'pixel_values_videos' and 'video_grid_thw'
            return_outputs: whether to return model outputs
            num_items_in_batch: number of items in batch (optional, for compatibility with Trainer)
        """
        pixel_values_videos = inputs['pixel_values_videos']
        video_grid_thw = inputs['video_grid_thw']
        
        # Forward pass
        loss, pred, mask = model(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mask_ratio=self.mask_ratio
        )
        
        return (loss, {'pred': pred, 'mask': mask}) if return_outputs else loss


def train_mae(args):
    # Create processor
    print(f"Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Create model
    print("Creating MAE model...")
    model = QwenViTMAE(
        model_id=args.model_id,
        decoder_embed_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_heads,
        mlp_ratio=args.mlp_ratio,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        spacetime_mask=args.spacetime_mask,
        mask_strategy=args.mask_strategy,
        mask_unit_size=tuple(args.mask_unit_size),  # Convert list to tuple
    )
    
    # Move model to device (DeepSpeed will handle multi-GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.deepspeed:
        model = model.to(device)
    
    # Validate encoder training configuration
    if args.freeze_encoder and args.unfreeze_topk_vision > 0:
        raise ValueError(
            "Cannot set both --freeze_encoder and --unfreeze_topk_vision. "
            "Use --freeze_encoder to freeze all layers, or --unfreeze_topk_vision to train only top k layers."
        )
    
    # Configure vision encoder training
    # Option 1: Freeze all encoder layers
    if args.freeze_encoder:
        for param in model.visual.parameters():
            param.requires_grad = False
        print("Vision encoder frozen (all layers)")
    # Option 2: Unfreeze only top k layers
    elif args.unfreeze_topk_vision > 0:
        # First freeze all layers
        for param in model.visual.parameters():
            param.requires_grad = False
        # Then unfreeze top k layers
        if hasattr(model.visual, "blocks"):
            total_blocks = len(model.visual.blocks)
            k = min(args.unfreeze_topk_vision, total_blocks)
            for blk in model.visual.blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
            print(f"Vision encoder: {k}/{total_blocks} top layers trainable (layers {total_blocks-k+1}-{total_blocks})")
        else:
            print("Warning: model.visual.blocks not found, cannot unfreeze top k layers")
            print("Vision encoder fully frozen")
    # Option 3: Train all layers (default)
    else:
        print("Vision encoder trainable (all layers)")
    
    # Create dataset
    print(f"Loading dataset from {args.data_path}...")
    if args.video_base_path:
        print(f"Using video base path: {args.video_base_path}")
    data_module = make_mae_data_module(
        model_id=args.model_id,
        processor=processor,
        data_path=args.data_path,
        video_resized_width=args.video_resized_width,
        video_resized_height=args.video_resized_height,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        fps=args.fps,
        nframes=args.nframes,
        video_base_path=args.video_base_path,
    )
    
    train_dataset = data_module['dataset']
    data_collator = data_module['data_collator']
    
    # Calculate and print global training steps
    dataset_size = len(train_dataset)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.deepspeed and num_gpus == 0:
        # os is already imported at the top of the file
        world_size = int(os.environ.get('WORLD_SIZE', num_gpus if num_gpus > 0 else 2))
        num_gpus = world_size if world_size > 1 else 2  # Default to 2 for DeepSpeed
    
    per_device_batch_size = args.batch_size
    # Global steps per epoch (all GPUs combined)
    steps_per_epoch = (dataset_size + (per_device_batch_size * num_gpus) - 1) // (per_device_batch_size * num_gpus)
    global_total_steps = steps_per_epoch * args.num_epochs
    
    print(f"\nGlobal training steps: {global_total_steps} ({steps_per_epoch} steps/epoch × {args.num_epochs} epochs, {dataset_size} samples, {num_gpus} GPUs)\n")
    
    # Initialize decoder_pred before DeepSpeed wraps the model
    # This ensures all modules exist before DeepSpeed's parameter management kicks in
    print("Initializing decoder_pred with dummy forward...")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # Get a sample batch to determine patch_pixel_dim
        sample_batch = data_collator([train_dataset[0]])
        pixel_values_videos = sample_batch['pixel_values_videos'].to(device)
        video_grid_thw = sample_batch['video_grid_thw'].to(device)
        # Run a dummy forward to initialize decoder_pred
        _ = model(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mask_ratio=args.mask_ratio
        )
    model.train()
    print("decoder_pred initialized successfully")
    
    # Note: Gradient checkpointing will be enabled by Trainer automatically
    # if args.gradient_checkpointing is True, so we don't need to call it here
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_interval,
        save_steps=args.save_interval if args.save_strategy == 'steps' else None,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16 if args.fp16 else False,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True} if args.gradient_checkpointing else None,
        report_to=[],  # Disable wandb/tensorboard for MAE (empty list = no logging)
        remove_unused_columns=False,  # Keep pixel_values_videos and video_grid_thw
        # DeepSpeed config
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        # Custom MAE argument (will be accessed via args in compute_loss)
    )
    
    # Create trainer with memory clearing callback
    # Always add callback to help with memory pressure (works for both DeepSpeed and non-DeepSpeed)
    callbacks = [MemoryClearingCallback()]
    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        mask_ratio=args.mask_ratio,  # Pass mask_ratio to trainer
        callbacks=callbacks,  # Add memory clearing callback
    )
        
    # Resume from checkpoint if available
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    
    # Train
    print(f"Starting training for {args.num_epochs} epochs...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Save final encoder for downstream tasks (on main process only)
    if trainer.is_world_process_zero():
    print("Saving final encoder...")
    encoder_path = Path(args.output_dir) / 'mae_encoder_final.pth'
    torch.save(model.visual.state_dict(), encoder_path)
    print(f'Training completed! Encoder saved to {encoder_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE pre-training for Qwen2-VL')
    
    # Model parameters
    parser.add_argument('--model_id', default='Qwen/Qwen2-VL-7B-Instruct', type=str)
    parser.add_argument('--mask_ratio', default=0.85, type=float, help='Mask ratio (0.75 or 0.90)')
    parser.add_argument('--decoder_dim', default=512, type=int, help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', default=8, type=int, help='Decoder depth (number of transformer blocks)')
    parser.add_argument('--decoder_heads', default=16, type=int, help='Decoder number of attention heads')
    parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio in transformer blocks')
    parser.add_argument('--norm_pix_loss', default=True, type=bool, help='Normalize pixel loss by patch std')
    parser.add_argument('--spacetime_mask', default=True, type=bool, help='Use random spacetime mask (T, H, W dimensions)')
    parser.add_argument('--mask_strategy', default='tube', type=str, choices=['random', 'tube', 'block', 'mu'],
                        help='Masking strategy: random (patch-level), tube (time-tube), block (spatial-block), mu (mask-unit like Hiera)')
    parser.add_argument('--mask_unit_size', default=[4, 4], type=int, nargs=2, metavar=('H', 'W'),
                        help='Mask unit size (H, W) for block and mu strategies. Default: 4 4')
    parser.add_argument('--freeze_encoder', default=False, type=bool, help='Freeze vision encoder (only train decoder). Mutually exclusive with --unfreeze_topk_vision')
    parser.add_argument('--unfreeze_topk_vision', default=0, type=int, help='Unfreeze only top k layers of vision encoder (0 = train all layers, mutually exclusive with --freeze_encoder)')
    
    # Training parameters
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=1.5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    
    # Data parameters
    parser.add_argument('--data_path', required=True, type=str, help='Path to JSON file containing video paths')
    parser.add_argument('--video_base_path', type=str, default=None, help='Base path for video files (used when JSON contains relative paths)')
    parser.add_argument('--video_resized_width', default=224, type=int)
    parser.add_argument('--video_resized_height', default=224, type=int)
    parser.add_argument('--video_min_pixels', default=224*224, type=int)
    parser.add_argument('--video_max_pixels', default=224*224, type=int)
    parser.add_argument('--fps', type=int, help='Frames per second (optional)')
    parser.add_argument('--nframes', type=int, help='Number of frames (optional, deprecated, use fps instead)')
    
    # Output parameters
    parser.add_argument('--output_dir', default='./mae_checkpoints', type=str)
    parser.add_argument('--save_strategy', default='steps', type=str, choices=['steps', 'epoch', 'no'], help='Save strategy')
    parser.add_argument('--save_total_limit', default=2, type=int, help='Limit number of checkpoints')
    
    # Other parameters
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--save_interval', default=1000, type=int, help='Save every N steps (if save_strategy=steps)')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 precision')
    parser.add_argument('--fp16', action='store_true', help='Use float16 precision (mutually exclusive with bf16)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing for memory efficiency')
    
    # DeepSpeed parameters
    parser.add_argument('--deepspeed', type=str, default=None, help='Path to DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_mae(args)
