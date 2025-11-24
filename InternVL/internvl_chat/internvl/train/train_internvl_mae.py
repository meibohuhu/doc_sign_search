#!/usr/bin/env python3
"""
MAE pre-training script for InternVL vision encoder
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
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
import logging

logger = logging.getLogger(__name__)

# Try to import dist for distributed training
try:
    import torch.distributed as dist
except ImportError:
    dist = None

# Import MAE modules
# Add the train directory to sys.path to enable direct imports
train_dir = os.path.dirname(os.path.abspath(__file__))
if train_dir not in sys.path:
    sys.path.insert(0, train_dir)

# Import from same directory
from internvl_mae import InternViTMAE
from internvl_mae_dataset import make_mae_data_module


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
    
    def _patch_no_sync_for_zero(self):
        """Patch accelerator's no_sync method to handle ZeRO Stage 2/3.
        
        ZeRO Stage 2 and 3 are incompatible with no_sync during gradient accumulation
        because they partition gradients. The accelerator's no_sync method is a 
        contextmanager that internally calls DeepSpeed engine's no_sync, which fails.
        We patch it to directly return nullcontext() without calling any DeepSpeed methods.
        """
        # Only patch if DeepSpeed is enabled
        if not self.is_deepspeed_enabled:
            return
        
        # Check if already patched
        if hasattr(self.accelerator, '_internvl_no_sync_patched'):
            return
        
        import contextlib
        from contextlib import contextmanager
        
        # Store original no_sync method
        if not hasattr(self.accelerator, '_original_no_sync'):
            self.accelerator._original_no_sync = self.accelerator.no_sync
        
        # Create a patched no_sync that always returns nullcontext for ZeRO
        @contextmanager
        def no_sync_patched(model):
            """Patched no_sync that returns nullcontext for ZeRO Stage 2/3.
            
            For ZeRO Stage 2/3, return nullcontext instead of trying to use no_sync.
            This bypasses the DeepSpeed no_sync assertion error.
            """
            # Just yield - this creates a no-op context manager
            yield
        
        # Replace the method directly
        # The contextmanager decorator creates a function that returns a context manager
        self.accelerator.no_sync = no_sync_patched
        
        # Mark as patched
        self.accelerator._internvl_no_sync_patched = True
        
        # Log patch application (only on main process)
        try:
            if dist is not None and dist.is_initialized() and dist.get_rank() == 0:
                logger.info("✅ Patched accelerator.no_sync for DeepSpeed ZeRO Stage 2/3 compatibility")
            elif dist is None or not dist.is_initialized():
                logger.info("✅ Patched accelerator.no_sync for DeepSpeed ZeRO Stage 2/3 compatibility")
        except:
            pass
    
    def __init__(self, mask_ratio=0.75, **kwargs):
        """Initialize MAETrainer with mask_ratio"""
        self.mask_ratio = mask_ratio
        super().__init__(**kwargs)
        # Patch accelerator after initialization if DeepSpeed is enabled
        if hasattr(self, 'accelerator') and self.is_deepspeed_enabled:
            self._patch_no_sync_for_zero()
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Override train method to ensure patch is applied before training starts."""
        # Force patch before training starts (after model preparation)
        if hasattr(self, 'accelerator') and self.is_deepspeed_enabled:
            self._patch_no_sync_for_zero()
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, 
                            ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)
    
    def training_step(self, model, inputs):
        """Override training_step to apply patch before calling super()."""
        # CRITICAL: Patch must be applied BEFORE calling super().training_step()
        # because super() will call accelerator.accumulate() which calls no_sync()
        if hasattr(self, 'accelerator') and self.is_deepspeed_enabled:
            self._patch_no_sync_for_zero()
        return super().training_step(model, inputs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute MAE reconstruction loss
        
        Args:
            model: InternViTMAE model
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
    
    ### mh: 2025-11-24: Rotate checkpoints to enforce save_total_limit
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override _save_checkpoint to ensure save_total_limit is enforced.
        This ensures old checkpoints are deleted according to save_total_limit setting.
        """
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        
        # Get output directory before calling parent method
        run_dir = self._get_output_dir(trial=trial)
        checkpoint_saved = False
        
        # Call parent method to save checkpoint
        try:
            super()._save_checkpoint(model, trial, metrics=metrics)
            checkpoint_saved = True
        except Exception as e:
            # Log error but don't fail if checkpoint save had issues
            logger.warning(f"Checkpoint save encountered an issue: {e}")
            # Check if checkpoint directory was created despite the error
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(run_dir, checkpoint_folder)
            if os.path.exists(output_dir):
                checkpoint_saved = True
        
        # Rotate checkpoints to enforce save_total_limit
        # This ensures old checkpoints are deleted according to save_total_limit setting
        # Only rotate if checkpoint was successfully saved
        if checkpoint_saved and self.args.should_save and hasattr(self.args, 'save_total_limit') and self.args.save_total_limit is not None and self.args.save_total_limit > 0:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)


def train_mae(args):
    # Create model
    print("Creating MAE model...")
    model = InternViTMAE(
        model_id=args.model_id,
        vision_path=args.vision_path,
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
        if hasattr(model.visual, "encoder") and hasattr(model.visual.encoder, "layers"):
            total_layers = len(model.visual.encoder.layers)
            k = min(args.unfreeze_topk_vision, total_layers)
            for layer in model.visual.encoder.layers[-k:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"Vision encoder: {k}/{total_layers} top layers trainable (layers {total_layers-k+1}-{total_layers})")
        else:
            print("Warning: model.visual.encoder.layers not found, cannot unfreeze top k layers")
            print("Vision encoder fully frozen")
    # Option 3: Train all layers (default)
    else:
        print("Vision encoder trainable (all layers)")
    
    # Create dataset
    print(f"Loading dataset from {args.data_path}...")
    if args.video_base_path:
        print(f"Using video base path: {args.video_base_path}")
    data_module = make_mae_data_module(
        data_path=args.data_path,
        image_size=args.image_size,
        min_num_frames=args.min_num_frames,
        max_num_frames=args.max_num_frames,
        sampling_method=args.sampling_method,
        video_base_path=args.video_base_path,
        transform=None,  # Will use default transform
    )
    
    train_dataset = data_module['dataset']
    data_collator = data_module['data_collator']
    
    # Calculate and print global training steps
    dataset_size = len(train_dataset)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.deepspeed and num_gpus == 0:
        world_size = int(os.environ.get('WORLD_SIZE', num_gpus if num_gpus > 0 else 2))
        num_gpus = world_size if world_size > 1 else 2  # Default to 2 for DeepSpeed
    
    per_device_batch_size = args.batch_size
    # Global steps per epoch (all GPUs combined)
    steps_per_epoch = (dataset_size + (per_device_batch_size * num_gpus) - 1) // (per_device_batch_size * num_gpus)
    global_total_steps = steps_per_epoch * args.num_epochs
    
    print(f"\nGlobal training steps: {global_total_steps} ({steps_per_epoch} steps/epoch × {args.num_epochs} epochs, {dataset_size} samples, {num_gpus} GPUs)\n")
    
    # Initialize decoder_pred before DeepSpeed wraps the model
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
        report_to=[],  # Disable wandb/tensorboard for MAE
        remove_unused_columns=False,  # Keep pixel_values_videos and video_grid_thw
        # DeepSpeed config
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
    )
    
    # Create trainer with memory clearing callback
    callbacks = [MemoryClearingCallback()]
    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        mask_ratio=args.mask_ratio,  # Pass mask_ratio to trainer
        callbacks=callbacks,
    )
    
    # CRITICAL: Force patch accelerator.no_sync after trainer initialization
    # This must be done after accelerator is fully initialized and before training starts
    if training_args.do_train and hasattr(trainer, 'accelerator') and trainer.is_deepspeed_enabled:
        logger.info("Patching accelerator.no_sync for DeepSpeed ZeRO Stage 2/3 compatibility...")
        trainer._patch_no_sync_for_zero()
        # Verify patch was applied
        if hasattr(trainer.accelerator, '_internvl_no_sync_patched'):
            logger.info("✅ Successfully patched accelerator.no_sync")
        else:
            logger.warning("⚠️  Patch may not have been applied correctly")
        
    # Resume from checkpoint if available
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    
    # Train with error handling
    print(f"Starting training for {args.num_epochs} epochs...")
    try:
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
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        if trainer.is_world_process_zero():
            print(f"Training stopped at step {trainer.state.global_step}")
        raise
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if trainer.is_world_process_zero():
            print(f"Training stopped at step {trainer.state.global_step}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE pre-training for InternVL')
    
    # Model parameters
    parser.add_argument('--model_id', default='OpenGVLab/InternVL2_5-2B', type=str, help='Model ID for loading config')
    parser.add_argument('--vision_path', type=str, default=None, help='Path to InternVL vision model (if different from model_id)')
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
    parser.add_argument('--image_size', default=224, type=int, help='Image size (will be resized to square)')
    parser.add_argument('--min_num_frames', default=8, type=int, help='Minimum number of frames')
    parser.add_argument('--max_num_frames', default=32, type=int, help='Maximum number of frames')
    parser.add_argument('--sampling_method', default='rand', type=str, help='Frame sampling method (rand, fpsX.X, random_start_every2)')
    
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

