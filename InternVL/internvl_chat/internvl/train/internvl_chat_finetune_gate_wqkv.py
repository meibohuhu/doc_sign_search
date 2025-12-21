# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Iterable, List, Literal, Optional

import numpy as np

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2_gate_wqkv import InternLM2ForCausalLM

# CRITICAL: Patch modeling_internvl_chat.py to use gate version before importing InternVLChatModel
# This ensures InternVLChatModel uses the gated attention implementation
import sys
import internvl.model.internvl_chat.modeling_internvl_chat as modeling_internvl_chat_module
# Replace the import in modeling_internvl_chat module
modeling_internvl_chat_module.InternLM2ForCausalLM = InternLM2ForCausalLM
print("✅ Patched modeling_internvl_chat to use InternLM2ForCausalLM from modeling_internlm2_gate")

from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader, get_frame_indices,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, get_frame_indices,
                                    preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset, get_worker_info
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

##### mhu added code 1108 #####
class InternVLTrainer(Trainer):
    def create_optimizer(self):
        # Use standard Trainer optimizer creation
        return super().create_optimizer()

### mh1122 ##### ERROR: [rank0]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2
### incompatibility between no_sync and gradient partitioning logic of ZeRO stage 2
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
            if dist.is_initialized() and dist.get_rank() == 0:
                logger.info("✅ Patched accelerator.no_sync for DeepSpeed ZeRO Stage 2/3 compatibility")
            elif not dist.is_initialized():
                logger.info("✅ Patched accelerator.no_sync for DeepSpeed ZeRO Stage 2/3 compatibility")
        except:
            pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

#####################################################
    def training_step(self, model, inputs):
        # mh 1119: CRITICAL: Patch must be applied BEFORE calling super().training_step()
        # because super() will call accelerator.accumulate() which calls no_sync()
        if hasattr(self, 'accelerator') and self.is_deepspeed_enabled:
            self._patch_no_sync_for_zero()
        ###################################################################################################################################
        result = super().training_step(model, inputs)
        # Clear CUDA cache after each step to reduce memory pressure
        # This helps prevent pytorch allocator cache flushes in DeepSpeed ZeRO-3
        # The warning suggests ensuring all ranks flush their caches at the same time
        if torch.cuda.is_available():
            try:
                # Try to use DeepSpeed's accelerator if available (recommended for ZeRO-3)
                from deepspeed import get_accelerator
                # Synchronize all ranks before clearing cache
                if dist.is_initialized():
                    dist.barrier()
                get_accelerator().empty_cache()
            except (ImportError, AttributeError):
                # Fallback to PyTorch's empty_cache if DeepSpeed accelerator not available
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.empty_cache()
        return result

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override _save_checkpoint to handle directory rename errors gracefully.
        This fixes the issue where tmp-checkpoint-{step} directory may be deleted
        or missing during rename operation in distributed training.
        """
        import shutil
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        
        # Get the staging and final output directories before calling parent method
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        # mh: 2025-11-24: Call parent method to save checkpoint
        checkpoint_saved = False
        try:
            super()._save_checkpoint(model, trial, metrics=metrics)
            checkpoint_saved = True
        #####################################################

        except FileNotFoundError as e:
            # Check if the error is about the staging directory not existing during rename
            if "tmp-checkpoint" in str(e) or staging_output_dir in str(e):
                # Check if the final checkpoint directory exists (may have been renamed by another process)
                if os.path.exists(output_dir):
                    # Checkpoint was successfully saved, just the rename failed
                    # This can happen in distributed training when another process already renamed it
                    logger.warning(f"Checkpoint save completed but rename failed. "
                                 f"Checkpoint exists at {output_dir}. "
                                 f"Original error: {e}")
                    # Don't re-raise, checkpoint is actually saved
                    checkpoint_saved = True
                else:
                    # Checkpoint save actually failed
                    logger.error(f"Checkpoint save failed: {e}")
                    raise
            else:
                # Some other FileNotFoundError, re-raise it
                raise
        
        # After parent method returns, verify the checkpoint was saved correctly
        # If staging directory still exists, try to rename it
        if os.path.exists(staging_output_dir) and not os.path.exists(output_dir):
            # Parent's rename may have failed silently, try our own rename
            try:
                # Ensure all processes are synchronized before rename
                if dist.is_initialized():
                    dist.barrier()
                os.rename(staging_output_dir, output_dir)
                logger.info(f"Successfully renamed checkpoint from {staging_output_dir} to {output_dir}")
            except (FileNotFoundError, OSError) as e:
                # If staging directory doesn't exist, check if final directory exists
                if os.path.exists(output_dir):
                    # Checkpoint was already renamed (possibly by another process)
                    logger.warning(f"Checkpoint directory {output_dir} already exists. "
                                 f"Staging directory {staging_output_dir} was missing. "
                                 f"This may indicate a race condition in distributed training.")
                else:
                    # Try using shutil.move as fallback (works across filesystems)
                    try:
                        shutil.move(staging_output_dir, output_dir)
                        logger.info(f"Successfully moved checkpoint from {staging_output_dir} to {output_dir} using shutil.move")
                    except Exception as e2:
                        logger.error(f"Failed to rename/move checkpoint directory: {e2}")
                        # Don't raise, checkpoint files may still be saved

        ### mh: 2025-11-24: Rotate checkpoints to enforce save_total_limit
        # Rotate checkpoints to enforce save_total_limit
        # This ensures old checkpoints are deleted according to save_total_limit setting
        # Only rotate if checkpoint was successfully saved
        if checkpoint_saved and self.args.should_save and hasattr(self.args, 'save_total_limit') and self.args.save_total_limit is not None and self.args.save_total_limit > 0:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

###########

##### mhu added code 1122 #####
def _compute_frame_indices(
    sample: str,
    vlen: int,
    input_fps: float,
    max_num_frames: int,
    min_num_frames: int,
    start_index: int = 0,
) -> List[int]:
    """
    Compute frame indices based on sampling strategy.
    
    Args:
        sample: Sampling strategy. Supports:
            - 'fpsX.X': FPS-based sampling (e.g., 'fps2.0', 'fps12.0')
            - 'random_start_every2': Random start frame, then sample every 2 frames
            - Other strategies can be added here
        vlen: Video length (number of frames in the clip)
        input_fps: Original video FPS
        max_num_frames: Maximum number of frames to sample
        min_num_frames: Minimum number of frames to sample
        start_index: Starting frame index offset (for clip parameter)
    
    Returns:
        List of frame indices (relative to start_index)
    """
    frame_indices: List[int] = []
    
    if 'fps' in sample:
        # FPS-based sampling (same as InternVL's get_frame_indices)
        # Format: 'fpsX.X' where X.X is the target FPS
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps if input_fps > 0 else 0
        delta = 1 / output_fps  # gap between frames
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int).tolist()
        frame_indices = [e for e in frame_indices if e < vlen]
        
        # Apply max_num_frames limit: uniformly drop some to maintain even distribution
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            indices_to_keep = np.linspace(0, len(frame_indices) - 1, max_num_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices_to_keep]
    
    elif sample == 'random_start_every2':
        # Randomly choose a start frame, then sample every 2 frames
        if vlen > 0:
            # Randomly choose starting position from [0, vlen-1]
            start_offset = np.random.randint(0, max(1, vlen))
            # Sample every 2 frames starting from start_offset
            frame_indices = list(range(start_offset, vlen, 2))
            
            # If we have more frames than max_num_frames, truncate from the end
            if len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
    
    else:
        # Default: Sample every other frame (take 1 frame out of every 2 frames)
        # Randomly choose starting position (0 or 1) for better diversity
        start_offset = np.random.randint(0, 2)  # 0 or 1
        frame_indices = list(range(start_offset, vlen, 2))  # [0,2,4,...] or [1,3,5,...]
        
        # If we have more frames than max_num_frames, uniformly drop some to maintain even distribution
        if len(frame_indices) > max_num_frames:
            indices_to_keep = np.linspace(0, len(frame_indices) - 1, max_num_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices_to_keep]
    
    # Ensure we have at least min_num_frames if video has enough frames
    if len(frame_indices) < min_num_frames and vlen >= min_num_frames:
        # If we don't have enough, uniformly sample additional frames from the remaining frames
        remaining_indices = [i for i in range(vlen) if i not in frame_indices]
        needed = min_num_frames - len(frame_indices)
        if len(remaining_indices) >= needed:
            # Uniformly sample from remaining frames
            if len(remaining_indices) == needed:
                additional = remaining_indices
            else:
                indices_to_add = np.linspace(0, len(remaining_indices) - 1, needed, dtype=int)
                additional = [remaining_indices[i] for i in indices_to_add]
            frame_indices = sorted(frame_indices + additional)
    
    # Adjust indices if start_index is specified (for clip parameter)
    if start_index > 0:
        frame_indices = [f + start_index for f in frame_indices]
    
    # Remove duplicates and sort
    frame_indices = sorted(list(set(frame_indices)))
    return frame_indices


def _load_video_locally(
    video_path: str,
    max_num_frames: int,
    min_num_frames: int,
    sample: str = 'rand',
    clip: Optional[Iterable[int]] = None,
) -> List[Image.Image]:
    """
    Load video frames locally using decord or OpenCV as fallback.
    Supports multiple sampling strategies:
    - 'fpsX.X': FPS-based sampling (e.g., 'fps2.0', 'fps12.0')
    - 'random_start_every2': Random start frame, then sample every 2 frames
    - Default: Sample every other frame with random start (0 or 1)
    """
    load_errors: List[str] = []

    # Try decord first
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / float(fps) if fps > 0 else 0
        
        # Handle clip parameter (same as read_frames_decord)
        if clip and len(clip) == 2:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps) if fps > 0 else total_frames
            start_index = int(start * fps) if fps > 0 else 0
        else:
            vlen = total_frames
            start_index = 0

        # Compute frame indices based on sampling strategy
        frame_indices = _compute_frame_indices(
            sample=sample,
            vlen=vlen,
            input_fps=fps,
            max_num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            start_index=start_index,
        )
        
        # Clip indices to valid range
        frame_indices = [min(max(int(idx), 0), total_frames - 1) for idx in frame_indices]
        
        # Remove duplicates after clipping
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices
        
        frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
        images = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return images
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f'decord: {exc}')

    # Fallback to OpenCV
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('cv2.VideoCapture failed to open file')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        if total_frames == 0:
            raise RuntimeError('cv2 reported zero frames')

        # Handle clip parameter
        if clip and len(clip) == 2:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps)
            start_index = int(start * fps)
        else:
            vlen = total_frames
            start_index = 0

        # Compute frame indices based on sampling strategy
        frame_indices = _compute_frame_indices(
            sample=sample,
            vlen=vlen,
            input_fps=fps,
            max_num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            start_index=start_index,
        )
        
        # Clip indices to valid range
        frame_indices = [min(max(int(idx), 0), total_frames - 1) for idx in frame_indices]
        
        # Remove duplicates after clipping
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices

        # Read frames using OpenCV
        frames: List[Image.Image] = []
        frame_set = set(frame_indices)
        retrieved = 0
        
        for frame_idx in range(total_frames):
            success, frame = cap.read()
            if not success:
                break
            if frame_idx in frame_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                retrieved += 1
                if retrieved >= len(frame_indices):
                    break

        cap.release()

        if len(frames) == 0:
            raise RuntimeError('cv2 did not return any frames')

        return frames
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f'opencv: {exc}')

    error_msgs = '; '.join(load_errors) or 'unknown'
    raise RuntimeError(f'Failed to load video locally ({error_msgs})')


def _load_video_locally_1(
    video_path: str,
    max_num_frames: int,
    min_num_frames: int,
    sample: str = 'rand',
    clip: Optional[Iterable[int]] = None,
) -> List[Image.Image]:
    """
    Load video frames locally using read_frames_decord logic as default.
    This method uses the same frame sampling logic as read_frames_decord from dataset.py.
    Supports sampling strategies: 'rand', 'middle', 'fpsX.X'
    """
    from decord import VideoReader
    
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps) if fps > 0 else 0
    
    # Handle clip parameter (same as read_frames_decord)
    if clip and len(clip) == 2:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps) if fps > 0 else vlen
        start_index = int(start * fps) if fps > 0 else 0
    else:
        start_index = 0

    # Randomly select number of frames (same as read_frames_decord)
    t_num_frames = np.random.randint(min_num_frames, max_num_frames + 1)
    
    # Get frame indices using get_frame_indices (same as read_frames_decord)
    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=None,
        input_fps=fps, max_num_frames=max_num_frames
    )
    
    # Adjust indices if clip is specified (same as read_frames_decord)
    if clip and len(clip) == 2:
        frame_indices = [f + start_index for f in frame_indices]
    
    # Clip indices to valid range
    frame_indices = [min(max(int(idx), 0), len(video_reader) - 1) for idx in frame_indices]
    
    # Remove duplicates after clipping
    seen = set()
    unique_indices = []
    for idx in frame_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    frame_indices = unique_indices
    print(f'frame_indices: {frame_indices}')
    # Get frames using get_batch (same as read_frames_decord)
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    images = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return images
###################################################################################


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )
    headwise_attn_output_gate: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable headwise attention output gating. Default is False.'}
    )
    elementwise_attn_output_gate: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable elementwise attention output gating. Default is False.'}
    )
    qkv_bias: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable bias in Q/K/V/O projection layers. Default is False.'}
    )
    visual_summary_head_gate: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable visual-conditioned head gate. Default is False.'}
    )
    visual_gate_mode: str = field(
        default='add_logits',
        metadata={'help': 'Visual gate combination mode: add_logits, mul, or replace. Default is add_logits.'}
    )
    visual_summary_aggregation: str = field(
        default='mean',
        metadata={'help': 'Visual summary aggregation method: mean, max, sum, or first. Default is mean.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
     # mhu 1122 added code
    sampling_method: str = field(
        default='rand',
        metadata={
            'help': (
                'Video frame sampling method. Options: '
                "'rand' (default: every 2 frames with random start 0/1), "
                "'fpsX.X' (FPS-based sampling, e.g., 'fps2.0', 'fps12.0'), "
                "'random_start_every2' (random start frame, then every 2 frames). "
                'Default is rand.'
            )
        },
    )
    ##########################################################################
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
        ### mhu 11/10 added
        annotation_path = meta['annotation']
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                self.raw_data = f.readlines()
        except UnicodeDecodeError:
            logger.warning(f'Failed to read {annotation_path} with utf-8; falling back to latin-1 with replacement.')
            with open(annotation_path, 'r', encoding='latin-1', errors='replace') as f:
                self.raw_data = f.readlines()
        ###### 
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)
        # print(f"video_path here: {video_path}")
        ## mhu added code 1108 #####
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     worker_info = get_worker_info()
            # if worker_info is None or worker_info.id == 0:
            #     print(f"video_path here: {video_path}")


        #  mhu 11/13 added code ##### Load the video frames using tcs_loader when available, otherwise fall back to local decoding
        if self.tcs_loader is not None:
            image_list = self.tcs_loader(
                video_path,
                image_type='video',
                max_num_frames=self.max_num_frame,
                min_num_frames=self.min_num_frame,
                sample=self.sampling_method,
                clip=data_item.get('clip', None))
        else:
            image_list = _load_video_locally(
                video_path,
                max_num_frames=self.max_num_frame,
                min_num_frames=self.min_num_frame,
                sample=self.sampling_method,
                clip=data_item.get('clip', None),
            )

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    sampling_method='rand',
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            sampling_method=sampling_method,   ##### mh 1122 added code
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
        print(f'train_dataset after packed: PackedDataset with {len(datasets)} source datasets, '
              f'num_images_expected={data_args.num_images_expected}, '
              f'max_packed_tokens={data_args.max_packed_tokens}')
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def main():
    #### mh1122: set ACCELERATE_GRADIENT_ACCUMULATION_STEPS to 1 to avoid memory leak
    import os  # Ensure os is available in this scope
    os.environ['ACCELERATE_GRADIENT_ACCUMULATION_STEPS'] = '1'

    # Verify that modeling_internvl_chat is using the gate version
    import internvl.model.internvl_chat.modeling_internvl_chat as modeling_internvl_chat_module
    if modeling_internvl_chat_module.InternLM2ForCausalLM.__module__ == 'internvl.model.internlm2.modeling_internlm2_gate_wqkv':
        logger.info("✅ Verified: modeling_internvl_chat is using InternLM2ForCausalLM from modeling_internlm2_gate_wqkv")
    else:
        logger.warning(f"⚠️  WARNING: modeling_internvl_chat is using {modeling_internvl_chat_module.InternLM2ForCausalLM.__module__}, not the gate version!")

    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    # Initialize TCSLoader if petrel_client is available and config file exists
    tcs_loader = None
    if has_tcs_loader:
        conf_path = os.path.expanduser('~/petreloss.conf')
        if os.path.exists(conf_path):
            try:
                tcs_loader = TCSLoader(conf_path)
                logger.info(f'TCSLoader initialized successfully with config: {conf_path}')
            except Exception as e:
                logger.warning(f'Failed to initialize TCSLoader: {e}. Will use local file loading.')
                tcs_loader = None
        else:
            logger.warning(f'TCSLoader config file not found: {conf_path}. Will use local file loading.')
    else:
        logger.info('petrel_client not installed. Will use local file loading.')

    # Patch Qwen2 attention classes for gate support (before packed training patch)
    if model_args.headwise_attn_output_gate or model_args.elementwise_attn_output_gate:
        try:
            from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES
            from internvl.model.qwen2.modeling_qwen2_gate import (
                Qwen2Attention as Qwen2AttentionGate,
                Qwen2FlashAttention2 as Qwen2FlashAttention2Gate,
                Qwen2SdpaAttention as Qwen2SdpaAttentionGate,
            )
            # Replace Qwen2 attention classes with gate versions
            QWEN2_ATTENTION_CLASSES['eager'] = Qwen2AttentionGate
            QWEN2_ATTENTION_CLASSES['flash_attention_2'] = Qwen2FlashAttention2Gate
            QWEN2_ATTENTION_CLASSES['sdpa'] = Qwen2SdpaAttentionGate
            logger.info('✅ Patched Qwen2 attention classes with gate versions')
        except ImportError as e:
            logger.warning(f'⚠️  Failed to patch Qwen2 attention classes: {e}')

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from internvl.patch import apply_liger_kernel_to_internvit
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
            # Phase 1: Standard Gated Attention
            if model_args.headwise_attn_output_gate or model_args.elementwise_attn_output_gate:
                config.llm_config.headwise_attn_output_gate = model_args.headwise_attn_output_gate
                config.llm_config.elementwise_attn_output_gate = model_args.elementwise_attn_output_gate
                gate_type = "headwise" if model_args.headwise_attn_output_gate else "elementwise"
                logger.info(f'✅ Enabled {gate_type} attention output gating')
                
            # Visual-conditioned gated attention
            if model_args.visual_summary_head_gate:
                config.llm_config.visual_summary_head_gate = model_args.visual_summary_head_gate
                config.llm_config.visual_gate_mode = model_args.visual_gate_mode
                config.llm_config.visual_summary_aggregation = model_args.visual_summary_aggregation
                logger.info(f'✅ Enabled visual-conditioned head gate with mode: {model_args.visual_gate_mode}, aggregation: {model_args.visual_summary_aggregation}')
                logger.info(f'[DEBUG] config.llm_config.visual_summary_head_gate = {config.llm_config.visual_summary_head_gate}')
            # InternLM2 uses config.bias (not qkv_bias) - no need to set qkv_bias
            # config.bias is already set from the pretrained model config
        elif config.llm_config.model_type == 'qwen2':   ### for 1B or 4B model
            config.llm_config._attn_implementation = 'flash_attention_2'  # for Qwen2
            logger.info('Using flash_attention_2 for Qwen2')
            # Phase 1: Standard Gated Attention for Qwen2
            if model_args.headwise_attn_output_gate or model_args.elementwise_attn_output_gate:
                config.llm_config.headwise_attn_output_gate = model_args.headwise_attn_output_gate
                config.llm_config.elementwise_attn_output_gate = model_args.elementwise_attn_output_gate
                gate_type = "headwise" if model_args.headwise_attn_output_gate else "elementwise"
                logger.info(f'✅ Enabled {gate_type} attention output gating for Qwen2')
            # Set qkv_bias
            config.llm_config.qkv_bias = model_args.qkv_bias
            logger.info(f'✅ Set qkv_bias to {model_args.qkv_bias} for Qwen2')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
            # Warn if gating is enabled for non-InternLM2/Qwen2 models
            if model_args.headwise_attn_output_gate or model_args.elementwise_attn_output_gate:
                logger.warning(
                    f'⚠️  Gated attention is only supported for InternLM2 and Qwen2 models. '
                    f'Current model type: {config.llm_config.model_type}. '
                    f'Gating parameters will be ignored.'
                )
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        if config.downsample_ratio != data_args.down_sample_ratio:
            logger.info('Overriding config.downsample_ratio (%s) with data_args.down_sample_ratio (%s)',
                        config.downsample_ratio, data_args.down_sample_ratio)
            config.downsample_ratio = data_args.down_sample_ratio
        # mh1211 Load model with ignore_mismatched_sizes if gating is enabled
        # Both InternLM2 and Qwen2 need ignore_mismatched when gating is enabled
        # InternLM2: wqkv.weight shape changes (adds gate_dim rows)
        # Qwen2: q_proj.weight shape changes (adds gate_dim rows)
        ignore_mismatched = (model_args.headwise_attn_output_gate or 
                            model_args.elementwise_attn_output_gate) and \
                            config.llm_config.model_type in ['internlm2', 'qwen2']
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config,
            ignore_mismatched_sizes=ignore_mismatched)
        
        # Ensure config is properly set after model loading
        if config.llm_config.model_type == 'internlm2':
            if hasattr(model.language_model, 'config'):
                model.language_model.config.visual_summary_head_gate = config.llm_config.visual_summary_head_gate
                model.language_model.config.visual_gate_mode = config.llm_config.visual_gate_mode
                model.language_model.config.visual_summary_aggregation = config.llm_config.visual_summary_aggregation
                # Update all attention modules' visual_summary_head_gate attribute
                if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
                    for layer in model.language_model.model.layers:
                        if hasattr(layer, 'attention'):
                            layer.attention.visual_summary_head_gate = config.llm_config.visual_summary_head_gate
                            layer.attention.visual_gate_mode = config.llm_config.visual_gate_mode
        
        # For InternLM2: wqkv.bias and wo.bias are controlled by config.bias (not qkv_bias)
        # config.bias is already set from the pretrained model config, so no manual initialization needed
        # For Qwen2: q_proj.bias, k_proj.bias, v_proj.bias, o_proj.bias (handled below)
        
        # Initialize q_proj/k_proj/v_proj/o_proj.bias for Qwen2
        # Note: q_proj/k_proj/v_proj always have bias=True (matching original Qwen2)
        # o_proj has bias only if qkv_bias=True
        # Bias values will be loaded from pretrained model in the manual loading section below
        if config.llm_config.model_type == 'qwen2':
            logger.info('Initializing q_proj/k_proj/v_proj/o_proj.bias for Qwen2 (will load from pretrained model)...')
            from internvl.model.qwen2.modeling_qwen2_gate import (
                Qwen2Attention as Qwen2AttentionGate,
                Qwen2FlashAttention2 as Qwen2FlashAttention2Gate,
                Qwen2SdpaAttention as Qwen2SdpaAttentionGate,
            )
            import contextlib
            try:
                from deepspeed import zero
                gather_context = zero.GatheredParameters
            except ImportError:
                gather_context = contextlib.nullcontext
            
            # Access model layers (handle PEFT wrapping if needed)
            try:
                from peft import PeftModel
                if isinstance(model.language_model, PeftModel):
                    actual_model = model.language_model.base_model.model
                else:
                    actual_model = model.language_model.model
            except (ImportError, AttributeError):
                actual_model = model.language_model.model
            
            bias_init_count = 0
            for i, layer in enumerate(actual_model.layers):
                attn = layer.self_attn
                if isinstance(attn, (Qwen2AttentionGate, Qwen2FlashAttention2Gate, Qwen2SdpaAttentionGate)):
                    with gather_context([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias]):
                        with torch.no_grad():
                            # Initialize q_proj/k_proj/v_proj bias to 0 (will be loaded from pretrained model)
                            # Note: These biases exist in pretrained model and will be loaded later
                            for bias_name, bias_param in [
                                ('q_proj.bias', attn.q_proj.bias),
                                ('k_proj.bias', attn.k_proj.bias),
                                ('v_proj.bias', attn.v_proj.bias),
                            ]:
                                if bias_param is not None:
                                    if torch.isnan(bias_param).any() or torch.isinf(bias_param).any():
                                        bias_param.zero_()
                                        logger.warning(f'Layer {i}: {bias_name} contained NaN/Inf! Initialized to 0.')
                                    else:
                                        bias_param.zero_()
                                    bias_init_count += 1
                    
                    # Initialize o_proj.bias only if qkv_bias is enabled
                    if config.llm_config.qkv_bias and attn.o_proj.bias is not None:
                        with gather_context([attn.o_proj.bias]):
                            with torch.no_grad():
                                if torch.isnan(attn.o_proj.bias).any() or torch.isinf(attn.o_proj.bias).any():
                                    attn.o_proj.bias.zero_()
                                    logger.warning(f'Layer {i}: o_proj.bias contained NaN/Inf! Initialized to 0.')
                                else:
                                    attn.o_proj.bias.zero_()
                                bias_init_count += 1
            
            if bias_init_count > 0:
                logger.info(f'✅ Initialized q_proj/k_proj/v_proj bias for {bias_init_count // 3} layers (Qwen2). Will load from pretrained model.')
        
        # Helper function to load safetensors (handles both single file and sharded(4B/8B))
        def load_safetensors_state_dict(model_path_or_dir):
            """
            Load safetensors state dict, handling both single file and sharded formats.
            
            Returns:
                tuple: (state_dict, base_dir) or (None, None) if not found
            """
            from safetensors.torch import load_file
            import json
            
            # Try to find the model directory
            model_dir = None
            if os.path.exists(model_path_or_dir) and os.path.isdir(model_path_or_dir):
                model_dir = model_path_or_dir
            else:
                # Try HuggingFace cache
                try:
                    from huggingface_hub import snapshot_download
                    model_dir = snapshot_download(
                        repo_id=model_path_or_dir,
                        local_files_only=True,
                        cache_dir=None
                    )
                except Exception:
                    try:
                        from huggingface_hub.constants import HF_HUB_CACHE
                        cache_model_name = model_path_or_dir.replace('/', '--')
                        cache_base = os.path.join(HF_HUB_CACHE, f'models--{cache_model_name}')
                        if os.path.exists(cache_base):
                            snapshots_dir = os.path.join(cache_base, 'snapshots')
                            if os.path.exists(snapshots_dir):
                                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                                if snapshots:
                                    model_dir = os.path.join(snapshots_dir, snapshots[0])
                    except Exception:
                        pass
            
            if model_dir is None:
                return None, None
            
            # Check for sharded safetensors (model.safetensors.index.json)
            index_file = os.path.join(model_dir, 'model.safetensors.index.json')
            if os.path.exists(index_file):
                logger.info(f'Found sharded safetensors index: {index_file}')
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                
                # Load all shards
                state_dict = {}
                weight_map = index_data.get('weight_map', {})
                shard_files = set(weight_map.values())
                
                logger.info(f'Loading {len(shard_files)} shard files...')
                for shard_file in shard_files:
                    shard_path = os.path.join(model_dir, shard_file)
                    if os.path.exists(shard_path):
                        logger.debug(f'Loading shard: {shard_file}')
                        shard_dict = load_file(shard_path)
                        state_dict.update(shard_dict)
                    else:
                        logger.warning(f'Shard file not found: {shard_path}')
                        return None, None
                
                logger.info(f'✅ Loaded {len(state_dict)} keys from {len(shard_files)} shards')
                return state_dict, model_dir
            
            # Try single file
            single_file = os.path.join(model_dir, 'model.safetensors')
            if os.path.exists(single_file):
                logger.info(f'Found single safetensors file: {single_file}')
                state_dict = load_file(single_file)
                logger.info(f'✅ Loaded {len(state_dict)} keys from single file')
                return state_dict, model_dir
            
            return None, None
        
        # mh1211 If ignore_mismatched_sizes was used, manually load base QKV weights from pretrained model
        # This is necessary because ignore_mismatched_sizes skips loading the entire wqkv.weight,
        # which causes the base QKV part to be randomly initialized (potentially with NaN/Inf)
        # For Qwen2, we need to manually load base q_proj weights (gate part is added to q_proj)
        if ignore_mismatched and config.llm_config.model_type == 'internlm2':
            logger.info('Manually loading base QKV weights from pretrained model (ignore_mismatched_sizes was used)...')
            try:
                pretrained_model_path = model_args.model_name_or_path
                pretrained_state_dict, model_dir = load_safetensors_state_dict(pretrained_model_path)
                
                if pretrained_state_dict is not None:
                    
                    from internvl.model.internlm2.modeling_internlm2_gate_wqkv import InternLM2Attention, InternLM2FlashAttention2
                    import contextlib
                    try:
                        from deepspeed import zero
                        gather_context = zero.GatheredParameters
                    except ImportError:
                        gather_context = contextlib.nullcontext
                    
                    # Debug: Print sample keys from pretrained_state_dict to verify key format
                    sample_keys = [k for k in pretrained_state_dict.keys() if 'wqkv' in k][:5]
                    logger.info(f'[DEBUG] Sample wqkv keys from pretrained_state_dict: {sample_keys}')
                    
                    loaded_count = 0
                    for i, layer in enumerate(model.language_model.model.layers):
                        attn = layer.attention
                        attn_type = type(attn).__name__
                        # Use getattr to safely access attributes (may not exist if model wasn't initialized with gating)
                        headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                        elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                        
                        # Debug: Log attention type and gate status
                        if i < 3:  # Only log first 3 layers to avoid spam
                            logger.info(f'[DEBUG] Layer {i}: attention type={attn_type}, headwise_gate={headwise_gate}, elementwise_gate={elementwise_gate}, isinstance={isinstance(attn, (InternLM2Attention, InternLM2FlashAttention2))}')
                        
                        if isinstance(attn, (InternLM2Attention, InternLM2FlashAttention2)):
                            if headwise_gate or elementwise_gate:
                                base_qkv_dim = (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim
                                # Construct the key name for this layer's wqkv.weight
                                key = f'language_model.model.layers.{i}.attention.wqkv.weight'
                                
                                # Debug: Log key lookup
                                if i < 3:
                                    logger.info(f'[DEBUG] Layer {i}: Looking for key: {key}, key exists: {key in pretrained_state_dict}')
                                
                                if key in pretrained_state_dict:
                                    pretrained_wqkv = pretrained_state_dict[key].to(dtype=torch.bfloat16)
                                    ### Base QKV 部分（前 4096 行）：来自预训练模型, 通过 copy_() 从预训练模型的 wqkv.weight 复制

                                    # Check if pretrained weight has correct shape [4096, 2048]
                                    if pretrained_wqkv.shape == (base_qkv_dim, attn.hidden_size):
                                        # Verify pretrained weight doesn't contain NaN/Inf
                                        if torch.isnan(pretrained_wqkv).any() or torch.isinf(pretrained_wqkv).any():
                                            logger.warning(
                                                f'Layer {i}: Pretrained wqkv.weight contains NaN/Inf! Skipping...'
                                            )
                                        else:
                                            # mh1212 Copy base QKV part (first base_qkv_dim rows) from pretrained
                                            # Use GatheredParameters for DeepSpeed ZeRO compatibility
                                            with gather_context([attn.wqkv.weight]):
                                                with torch.no_grad():
                                                    # Calculate gate_dim based on gating mode
                                                    if headwise_gate:
                                                        gate_dim = attn.num_heads
                                                    elif elementwise_gate:
                                                        gate_dim = attn.num_heads * attn.head_dim
                                                        # print(f'num_heads: {attn.num_heads}')  16
                                                        # print(f'head_dim: {attn.head_dim}') 128
                                                        # print(f'gate_dim: {gate_dim}') 2048
                                                    else:
                                                        gate_dim = 0
                                                    
                                                    # Copy base QKV part
                                                    attn.wqkv.weight[:base_qkv_dim, :].copy_(pretrained_wqkv)  ### Base QKV 部分（前 4096 行）：来自预训练模型, 通过 copy_() 从预训练模型的 wqkv.weight 复制

                                                    # Verify and fix gate part if needed (should be properly initialized, but check for abnormalities)
                                                    if gate_dim > 0:
                                                        current_gate_part = attn.wqkv.weight[base_qkv_dim:, :]
                                                        gate_mean = current_gate_part.mean().item()
                                                        gate_std = current_gate_part.std().item()
                                                        has_nan = torch.isnan(current_gate_part).any().item()
                                                        has_inf = torch.isinf(current_gate_part).any().item()
                                                        
                                                        # If gate part looks wrong (all zeros, NaN/Inf, or abnormal values), reinitialize it
                                                        if has_nan or has_inf:
                                                            current_gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                                            logger.warning(
                                                                f'Layer {i}: Gate part contained NaN/Inf! Reinitialized.'
                                                            )
                                                        elif abs(gate_mean) < 1e-6 and gate_std < 1e-6:
                                                            # All zeros - reinitialize
                                                            current_gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                                            logger.warning(
                                                                f'Layer {i}: Gate part was all zeros! Reinitialized.'
                                                            )
                                                        elif abs(gate_mean) > 0.1 or gate_std > 0.1:
                                                            # Abnormal values (should be ~N(0, 0.02^2) for initializer_range=0.02)
                                                            # Reinitialize to ensure proper distribution
                                                            current_gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                                            logger.warning(
                                                                f'Layer {i}: Gate part had abnormal values (mean={gate_mean:.6f}, std={gate_std:.6f})! '
                                                                f'Reinitialized with std={config.llm_config.initializer_range}.'
                                                            )
                                                    
                                                # Verify copy was successful
                                                if torch.isnan(attn.wqkv.weight[:base_qkv_dim, :]).any() or \
                                                   torch.isinf(attn.wqkv.weight[:base_qkv_dim, :]).any():
                                                    logger.error(
                                                        f'Layer {i}: Base QKV contains NaN/Inf after copy! '
                                                        f'This should not happen.'
                                                    )
                                                else:
                                                    loaded_count += 1
                                    else:
                                        logger.warning(
                                            f'Layer {i}: Pretrained wqkv.weight shape {pretrained_wqkv.shape} '
                                            f'does not match expected base shape ({base_qkv_dim}, {attn.hidden_size})'
                                        )
                                else:
                                    logger.warning(f'Layer {i}: Key {key} not found in pretrained state dict')
                    
                    if loaded_count > 0:
                        logger.info(f'✅ Manually loaded base QKV weights for {loaded_count} layers')
                    else:
                        # Debug: Print detailed error information
                        total_layers = len(model.language_model.model.layers)
                        logger.error(f'❌ CRITICAL: No base QKV weights were loaded for InternLM2.')
                        logger.error(f'   Total layers: {total_layers}')
                        logger.error(f'   Pretrained state dict keys: {len(pretrained_state_dict)} keys loaded')
                        
                        # Check why loading failed for first few layers
                        for i in range(min(3, total_layers)):
                            layer = model.language_model.model.layers[i]
                            attn = layer.attention
                            attn_type = type(attn).__name__
                            headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                            elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                            key = f'language_model.model.layers.{i}.attention.wqkv.weight'
                            
                            logger.error(f'   Layer {i}:')
                            logger.error(f'     - Attention type: {attn_type}')
                            logger.error(f'     - Is gate attention: {isinstance(attn, (InternLM2Attention, InternLM2FlashAttention2))}')
                            logger.error(f'     - headwise_gate: {headwise_gate}, elementwise_gate: {elementwise_gate}')
                            logger.error(f'     - Key exists in state_dict: {key in pretrained_state_dict}')
                            if key in pretrained_state_dict:
                                logger.error(f'     - Key shape: {pretrained_state_dict[key].shape}')
                                logger.error(f'     - Expected base_qkv_dim: {(attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim}')
                        
                        raise RuntimeError( ##### remove fallback, 直接抛出异常
                            f'❌ CRITICAL: No base QKV weights were loaded for InternLM2. '
                            f'This will cause training to fail with high initial loss. '
                            f'Please check that pretrained weights are available at: {pretrained_model_path}. '
                            f'See error logs above for detailed debugging information.'
                        )
                else:
                    raise FileNotFoundError(   ##### remove fallback, 直接抛出异常
                        f'❌ CRITICAL: Pretrained weight file not found for InternLM2. '
                        f'Tried: {pretrained_model_path} (checked for model.safetensors and model.safetensors.index.json). '
                        f'Cannot proceed without pretrained weights. Please ensure the model weights are available.'
                    )
            except (FileNotFoundError, RuntimeError):
                # Re-raise these errors
                raise
            except Exception as e:
                raise RuntimeError(   ##### remove fallback, 直接抛出异常
                    f'❌ CRITICAL: Failed to manually load base QKV weights for InternLM2: {e}. '
                    f'This will cause training to fail. Please check the error above and ensure pretrained weights are available.'
                ) from e
        
        # Gate parameters are automatically initialized by _init_weights (same as Qwen3)
        ## 初始化：只在模型加载时进行一次（24 层）
        ## 训练：基于初始值通过梯度下降更新，不再重新初始化
        # No manual initialization needed - post_init() will call _init_weights for all modules
        # This uses normal_(mean=0.0, std=initializer_range) for weights and zero_() for bias
        if ignore_mismatched and config.llm_config.model_type == 'internlm2':
            logger.info(
                f'✅ Gate parameters will be automatically initialized by _init_weights '
                f'(same as Qwen3: normal(mean=0.0, std={config.llm_config.initializer_range}))'
            )
            # Verify gate part initialization after manual base QKV loading
            logger.info('Verifying gate part initialization...')
            from internvl.model.internlm2.modeling_internlm2_gate_wqkv import InternLM2Attention, InternLM2FlashAttention2
            import contextlib
            try:
                from deepspeed import zero
                gather_context = zero.GatheredParameters
            except ImportError:
                gather_context = contextlib.nullcontext
            
            verified_count = 0
            for i, layer in enumerate(model.language_model.model.layers):
                attn = layer.attention
                if isinstance(attn, (InternLM2Attention, InternLM2FlashAttention2)):
                    headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                    elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                    if headwise_gate or elementwise_gate:
                        base_qkv_dim = (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim
                        with gather_context([attn.wqkv.weight]):
                            gate_part = attn.wqkv.weight[base_qkv_dim:, :]
                            gate_mean = gate_part.mean().item()
                            gate_std = gate_part.std().item()
                            gate_min = gate_part.min().item()
                            gate_max = gate_part.max().item()
                            has_nan = torch.isnan(gate_part).any().item()
                            has_inf = torch.isinf(gate_part).any().item()
                            
                            if has_nan or has_inf:
                                logger.warning(
                                    f'Layer {i}: Gate part contains NaN/Inf! Reinitializing...'
                                )
                                # Reinitialize gate part
                                with torch.no_grad():
                                    gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                verified_count += 1
                            elif abs(gate_mean) < 1e-6 and gate_std < 1e-6:
                                logger.warning(
                                    f'Layer {i}: Gate part appears to be all zeros! '
                                    f'mean={gate_mean:.6f}, std={gate_std:.6f}, '
                                    f'range=[{gate_min:.6f}, {gate_max:.6f}]. '
                                    f'This indicates initialization failed. Reinitializing...'
                                )
                                # Reinitialize gate part
                                with torch.no_grad():
                                    gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                verified_count += 1
                            elif abs(gate_mean) > 0.1 or gate_std > 0.1:
                                # Abnormal values (should be ~N(0, 0.02^2) for initializer_range=0.02)
                                # Values like mean=63963136 or mean=-9.38e+28 are clearly wrong
                                logger.warning(
                                    f'Layer {i}: Gate part has abnormal values! '
                                    f'mean={gate_mean:.6f}, std={gate_std:.6f}, '
                                    f'range=[{gate_min:.6f}, {gate_max:.6f}]. '
                                    f'Reinitializing with std={config.llm_config.initializer_range}...'
                                )
                                # Reinitialize gate part
                                with torch.no_grad():
                                    gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                verified_count += 1
                            else:
                                logger.info(
                                    f'Layer {i}: Gate part initialized correctly. '
                                    f'mean={gate_mean:.6f}, std={gate_std:.6f}, '
                                    f'range=[{gate_min:.6f}, {gate_max:.6f}]'
                                )
                                verified_count += 1
            
            if verified_count > 0:
                logger.info(f'✅ Verified and fixed gate part initialization for {verified_count} layers')
        
        # mh1216: Qwen2 gate initialization - manually load base q_proj weights and initialize gate part
        if ignore_mismatched and config.llm_config.model_type == 'qwen2':
            logger.info('Manually loading base q_proj weights from pretrained model for Qwen2 (ignore_mismatched_sizes was used)...')
            try:
                from internvl.model.qwen2.modeling_qwen2_gate import (
                    Qwen2Attention as Qwen2AttentionGate,
                    Qwen2FlashAttention2 as Qwen2FlashAttention2Gate,
                    Qwen2SdpaAttention as Qwen2SdpaAttentionGate,
                )
                
                pretrained_model_path = model_args.model_name_or_path
                pretrained_state_dict, model_dir = load_safetensors_state_dict(pretrained_model_path)
                
                if pretrained_state_dict is not None:
                    logger.info(f'Loading base q_proj weights from pretrained model...')
                    
                    import contextlib
                    try:
                        from deepspeed import zero
                        gather_context = zero.GatheredParameters
                    except ImportError:
                        gather_context = contextlib.nullcontext
                    
                    loaded_count = 0
                    # Access model layers (handle PEFT wrapping if needed)
                    try:
                        from peft import PeftModel
                        if isinstance(model.language_model, PeftModel):
                            actual_model = model.language_model.base_model.model
                        else:
                            actual_model = model.language_model.model
                    except (ImportError, AttributeError):
                        actual_model = model.language_model.model
                    
                    for i, layer in enumerate(actual_model.layers):
                        attn = layer.self_attn
                        if isinstance(attn, (Qwen2AttentionGate, Qwen2FlashAttention2Gate, Qwen2SdpaAttentionGate)):
                            headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                            elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                            if headwise_gate or elementwise_gate:
                                # Calculate dimensions
                                base_q_dim = attn.num_heads * attn.head_dim  # Base query dimension
                                if headwise_gate:
                                    gate_dim = attn.num_heads
                                elif elementwise_gate:
                                    gate_dim = attn.num_heads * attn.head_dim
                                else:
                                    gate_dim = 0
                                
                                # Construct the key name for this layer's q_proj.weight
                                key = f'language_model.model.layers.{i}.self_attn.q_proj.weight'
                                if key in pretrained_state_dict:
                                    pretrained_q_proj = pretrained_state_dict[key].to(dtype=torch.bfloat16)
                                    
                                    # Check if pretrained weight has correct shape [base_q_dim, hidden_size]
                                    if pretrained_q_proj.shape == (base_q_dim, attn.hidden_size):
                                        # Verify pretrained weight doesn't contain NaN/Inf
                                        if torch.isnan(pretrained_q_proj).any() or torch.isinf(pretrained_q_proj).any():
                                            logger.warning(
                                                f'Layer {i}: Pretrained q_proj.weight contains NaN/Inf! Skipping...'
                                            )
                                        else:
                                            # Copy base q_proj part (first base_q_dim rows) from pretrained
                                            with gather_context([attn.q_proj.weight]):
                                                with torch.no_grad():
                                                    # Copy base q_proj part
                                                    attn.q_proj.weight[:base_q_dim, :].copy_(pretrained_q_proj)
                                                    
                                                    # Verify and fix gate part if needed
                                                    if gate_dim > 0:
                                                        current_gate_part = attn.q_proj.weight[base_q_dim:, :]
                                                        gate_mean = current_gate_part.mean().item()
                                                        gate_std = current_gate_part.std().item()
                                                        has_nan = torch.isnan(current_gate_part).any().item()
                                                        has_inf = torch.isinf(current_gate_part).any().item()
                                                        
                                                        # If gate part looks wrong, reinitialize it
                                                        if has_nan or has_inf:
                                                            current_gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                                            logger.warning(
                                                                f'Layer {i}: Gate part contained NaN/Inf! Reinitialized.'
                                                            )
                                                        elif abs(gate_mean) < 1e-6 and gate_std < 1e-6:
                                                            current_gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                                            logger.warning(
                                                                f'Layer {i}: Gate part was all zeros! Reinitialized.'
                                                            )
                                                        elif abs(gate_mean) > 0.1 or gate_std > 0.1:
                                                            current_gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                                            logger.warning(
                                                                f'Layer {i}: Gate part had abnormal values (mean={gate_mean:.6f}, std={gate_std:.6f})! '
                                                                f'Reinitialized with std={config.llm_config.initializer_range}.'
                                                            )
                                                    
                                                    # Verify copy was successful
                                                    if torch.isnan(attn.q_proj.weight[:base_q_dim, :]).any() or \
                                                       torch.isinf(attn.q_proj.weight[:base_q_dim, :]).any():
                                                        logger.error(
                                                            f'Layer {i}: Base q_proj contains NaN/Inf after copy! '
                                                            f'This should not happen.'
                                                        )
                                                    else:
                                                        loaded_count += 1
                                                    
                                                    # Load q_proj.bias from pretrained model (if exists)  otherwise loss high, not sure why MHU 1216
                                                    bias_key = f'language_model.model.layers.{i}.self_attn.q_proj.bias'
                                                    if bias_key in pretrained_state_dict and attn.q_proj.bias is not None:
                                                        pretrained_q_bias = pretrained_state_dict[bias_key].to(dtype=torch.bfloat16)
                                                        if pretrained_q_bias.shape == (base_q_dim,):
                                                            if not (torch.isnan(pretrained_q_bias).any() or torch.isinf(pretrained_q_bias).any()):
                                                                with gather_context([attn.q_proj.bias]):
                                                                    with torch.no_grad():
                                                                        # Copy base q_proj.bias (first base_q_dim elements)
                                                                        attn.q_proj.bias[:base_q_dim].copy_(pretrained_q_bias)
                                                                        # Initialize gate part of bias to 0 (if exists)
                                                                        if gate_dim > 0:
                                                                            attn.q_proj.bias[base_q_dim:].zero_()
                                                    
                                                    # Load k_proj.bias and v_proj.bias from pretrained model (if exists)
                                                    for proj_name, proj_layer in [('k_proj', attn.k_proj), ('v_proj', attn.v_proj)]:
                                                        if proj_layer.bias is not None:
                                                            proj_bias_key = f'language_model.model.layers.{i}.self_attn.{proj_name}.bias'
                                                            if proj_bias_key in pretrained_state_dict:
                                                                pretrained_proj_bias = pretrained_state_dict[proj_bias_key].to(dtype=torch.bfloat16)
                                                                expected_bias_dim = attn.num_key_value_heads * attn.head_dim
                                                                if pretrained_proj_bias.shape == (expected_bias_dim,):
                                                                    if not (torch.isnan(pretrained_proj_bias).any() or torch.isinf(pretrained_proj_bias).any()):
                                                                        with gather_context([proj_layer.bias]):
                                                                            with torch.no_grad():
                                                                                proj_layer.bias.copy_(pretrained_proj_bias)
                                    else:
                                        logger.warning(
                                            f'Layer {i}: Pretrained q_proj.weight shape {pretrained_q_proj.shape} '
                                            f'does not match expected base shape ({base_q_dim}, {attn.hidden_size})'
                                        )
                                else:
                                    logger.warning(f'Layer {i}: Key {key} not found in pretrained state dict')
                    
                    if loaded_count > 0:
                        logger.info(f'✅ Manually loaded base q_proj weights for {loaded_count} layers (Qwen2)')
                    else:
                        raise RuntimeError(   ##### remove fallback, 直接抛出异常
                            f'❌ CRITICAL: No base q_proj weights were loaded for Qwen2. '
                            f'This will cause training to fail with high initial loss. '
                            f'Please check that pretrained weights are available at: {pretrained_model_path}'
                        )
                else:
                    raise FileNotFoundError(   ##### remove fallback, 直接抛出异常
                        f'❌ CRITICAL: Pretrained weight file not found for Qwen2. '
                        f'Tried: {pretrained_model_path} (checked for model.safetensors and model.safetensors.index.json). '
                        f'Cannot proceed without pretrained weights. Please ensure the model weights are available.'
                    )
            except (FileNotFoundError, RuntimeError):
                # Re-raise these errors
                raise
            except Exception as e:
                raise RuntimeError(   ##### remove fallback, 直接抛出异常
                    f'❌ CRITICAL: Failed to manually load base q_proj weights for Qwen2: {e}. '
                    f'This will cause training to fail. Please check the error above and ensure pretrained weights are available.'
                ) from e
            
            # Verify and fix gate part initialization for Qwen2
            logger.info('Verifying gate part initialization for Qwen2...')
            from internvl.model.qwen2.modeling_qwen2_gate import (
                Qwen2Attention as Qwen2AttentionGate,
                Qwen2FlashAttention2 as Qwen2FlashAttention2Gate,
                Qwen2SdpaAttention as Qwen2SdpaAttentionGate,
            )
            import contextlib
            try:
                from deepspeed import zero
                gather_context = zero.GatheredParameters
            except ImportError:
                gather_context = contextlib.nullcontext
            
            # Access model layers (handle PEFT wrapping if needed)
            try:
                from peft import PeftModel
                if isinstance(model.language_model, PeftModel):
                    actual_model = model.language_model.base_model.model
                else:
                    actual_model = model.language_model.model
            except (ImportError, AttributeError):
                actual_model = model.language_model.model
            
            verified_count = 0
            for i, layer in enumerate(actual_model.layers):
                attn = layer.self_attn
                if isinstance(attn, (Qwen2AttentionGate, Qwen2FlashAttention2Gate, Qwen2SdpaAttentionGate)):
                    headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                    elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                    if headwise_gate or elementwise_gate:
                        base_q_dim = attn.num_heads * attn.head_dim
                        if headwise_gate:
                            gate_dim = attn.num_heads
                        elif elementwise_gate:
                            gate_dim = attn.num_heads * attn.head_dim
                        else:
                            gate_dim = 0
                        
                        with gather_context([attn.q_proj.weight]):
                            gate_part = attn.q_proj.weight[base_q_dim:, :]
                            gate_mean = gate_part.mean().item()
                            gate_std = gate_part.std().item()
                            gate_min = gate_part.min().item()
                            gate_max = gate_part.max().item()
                            has_nan = torch.isnan(gate_part).any().item()
                            has_inf = torch.isinf(gate_part).any().item()
                            
                            if has_nan or has_inf:
                                logger.warning(
                                    f'Layer {i}: Gate part contains NaN/Inf! Reinitializing...'
                                )
                                with torch.no_grad():
                                    gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                verified_count += 1
                            elif abs(gate_mean) < 1e-6 and gate_std < 1e-6:
                                logger.warning(
                                    f'Layer {i}: Gate part appears to be all zeros! '
                                    f'mean={gate_mean:.6f}, std={gate_std:.6f}, '
                                    f'range=[{gate_min:.6f}, {gate_max:.6f}]. Reinitializing...'
                                )
                                with torch.no_grad():
                                    gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                verified_count += 1
                            elif abs(gate_mean) > 0.1 or gate_std > 0.1:
                                logger.warning(
                                    f'Layer {i}: Gate part has abnormal values! '
                                    f'mean={gate_mean:.6f}, std={gate_std:.6f}, '
                                    f'range=[{gate_min:.6f}, {gate_max:.6f}]. '
                                    f'Reinitializing with std={config.llm_config.initializer_range}...'
                                )
                                with torch.no_grad():
                                    gate_part.normal_(mean=0.0, std=config.llm_config.initializer_range)
                                verified_count += 1
                            else:
                                logger.info(
                                    f'Layer {i}: Gate part initialized correctly. '
                                    f'mean={gate_mean:.6f}, std={gate_std:.6f}, '
                                    f'range=[{gate_min:.6f}, {gate_max:.6f}]'
                                )
                                verified_count += 1
            
            if verified_count > 0:
                logger.info(f'✅ Verified and fixed gate part initialization for {verified_count} layers (Qwen2)')
        
        elif ignore_mismatched and config.llm_config.model_type not in ['internlm2', 'qwen2']:
            logger.warning(
                f'⚠️  Gated attention is only supported for InternLM2 and Qwen2 models. '
                f'Current model type: {config.llm_config.model_type}. '
                f'Gating will not be applied.'
            )
        ############################
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
            # mh 1211 Phase 1: Standard Gated Attention
            if model_args.headwise_attn_output_gate or model_args.elementwise_attn_output_gate:
                llm_config.headwise_attn_output_gate = model_args.headwise_attn_output_gate
                llm_config.elementwise_attn_output_gate = model_args.elementwise_attn_output_gate
                gate_type = "headwise" if model_args.headwise_attn_output_gate else "elementwise"
                logger.info(f'✅ Enabled {gate_type} attention output gating')
            # Set qkv_bias
            llm_config.qkv_bias = model_args.qkv_bias
            logger.info(f'✅ Set qkv_bias to {model_args.qkv_bias}')
            # Visual-conditioned gated attention
            if model_args.visual_summary_head_gate:
                llm_config.visual_summary_head_gate = model_args.visual_summary_head_gate
                llm_config.visual_gate_mode = model_args.visual_gate_mode
                llm_config.visual_summary_aggregation = model_args.visual_summary_aggregation
                logger.info(f'✅ Enabled visual-conditioned head gate with mode: {model_args.visual_gate_mode}, aggregation: {model_args.visual_summary_aggregation}')
        elif llm_config.model_type == 'qwen2':
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for Qwen2
            logger.info('Using flash_attention_2 for Qwen2')
            # Phase 1: Standard Gated Attention for Qwen2
            if model_args.headwise_attn_output_gate or model_args.elementwise_attn_output_gate:
                llm_config.headwise_attn_output_gate = model_args.headwise_attn_output_gate
                llm_config.elementwise_attn_output_gate = model_args.elementwise_attn_output_gate
                gate_type = "headwise" if model_args.headwise_attn_output_gate else "elementwise"
                logger.info(f'✅ Enabled {gate_type} attention output gating for Qwen2')
            # Set qkv_bias
            llm_config.qkv_bias = model_args.qkv_bias
            logger.info(f'✅ Set qkv_bias to {model_args.qkv_bias} for Qwen2')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
            # Warn if gating is enabled for non-InternLM2/Qwen2 models
            if model_args.headwise_attn_output_gate or model_args.elementwise_attn_output_gate:
                logger.warning(
                    f'⚠️  Gated attention is only supported for InternLM2 and Qwen2 models. '
                    f'Current model type: {llm_config.model_type}. '
                    f'Gating parameters will be ignored.'
                )
        # Load LLM with ignore_mismatched_sizes if gating is enabled
        # Only use ignore_mismatched for InternLM2 (Qwen2 uses separate q_proj/k_proj/v_proj, so no mismatch)
        ignore_mismatched = (model_args.headwise_attn_output_gate or 
                            model_args.elementwise_attn_output_gate) and llm_config.model_type == 'internlm2'
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True,
            ignore_mismatched_sizes=ignore_mismatched)
        
        # For InternLM2: wqkv.bias and wo.bias are controlled by config.bias (not qkv_bias)
        # config.bias is already set from the pretrained model config, so no manual initialization needed
        
        # Gate parameters are automatically initialized by _init_weights (same as Qwen3)
        # No manual initialization needed - post_init() will call _init_weights for all modules
        # This uses normal_(mean=0.0, std=initializer_range) for weights and zero_() for bias
        if ignore_mismatched and llm_config.model_type == 'internlm2':
            logger.info(
                f'✅ Gate parameters will be automatically initialized by _init_weights '
                f'(same as Qwen3: normal(mean=0.0, std={llm_config.initializer_range}))'
            )
        elif ignore_mismatched and llm_config.model_type != 'internlm2':
            logger.warning(
                f'⚠️  Gated attention is only supported for InternLM2 models. '
                f'Current model type: {llm_config.model_type}. '
                f'Gating will not be applied.'
            )

                #######################
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id

    ### mhu 2025-11-08: add downsample_ratio check
    if model.config.downsample_ratio != data_args.down_sample_ratio:
        logger.warning('model.config.downsample_ratio (%s) != data_args.down_sample_ratio (%s); '
                       'updating model.config to match data_args',
                       model.config.downsample_ratio, data_args.down_sample_ratio)
        model.config.downsample_ratio = data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type, min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame, sampling_method=data_args.sampling_method) ###### mh 1122 added code

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora
        
        # Ensure bias parameters are trainable when using LoRA
        # Note: PEFT LoRA defaults to bias="none" (frozen), but since qkv_bias creates NEW bias
        # parameters (initialized to 0), we need to train them. If bias was part of pretrained
        # For InternLM2: wqkv.bias and wo.bias are controlled by config.bias (not qkv_bias)
        # If config.bias is True, bias parameters will be trainable automatically
        # For Qwen2: q_proj.bias, k_proj.bias, v_proj.bias, o_proj.bias (handled below)
        
        # Ensure Qwen2 bias parameters are trainable when using LoRA
        if model_args.qkv_bias and model.config.llm_config.model_type == 'qwen2' and model_args.use_llm_lora:
            logger.info('Ensuring q_proj/k_proj/v_proj/o_proj.bias are trainable with LoRA for Qwen2...')
            logger.info('  (These are NEW bias parameters initialized to 0, so they need training)')
            from internvl.model.qwen2.modeling_qwen2_gate import (
                Qwen2Attention as Qwen2AttentionGate,
                Qwen2FlashAttention2 as Qwen2FlashAttention2Gate,
                Qwen2SdpaAttention as Qwen2SdpaAttentionGate,
            )
            
            # Handle PEFT-wrapped model: get the actual model from base_model
            try:
                from peft import PeftModel
                if isinstance(model.language_model, PeftModel):
                    base_model = model.language_model.base_model
                    actual_model = base_model.model
                else:
                    actual_model = model.language_model.model
            except (ImportError, AttributeError) as e:
                logger.warning(f'Failed to access PEFT model structure: {e}, trying direct access')
                actual_model = model.language_model.model
            
            # Ensure we have the model with layers attribute
            if not hasattr(actual_model, 'layers'):
                if hasattr(actual_model, 'model'):
                    actual_model = actual_model.model
                else:
                    logger.error(f'actual_model type: {type(actual_model)}, attributes: {[a for a in dir(actual_model) if not a.startswith("_")][:10]}')
                    raise AttributeError(f'actual_model ({type(actual_model)}) does not have "layers" attribute')
            
            bias_trainable_count = 0
            for i, layer in enumerate(actual_model.layers):
                attn = layer.self_attn
                if isinstance(attn, (Qwen2AttentionGate, Qwen2FlashAttention2Gate, Qwen2SdpaAttentionGate)):
                    # Ensure all bias parameters are trainable
                    for bias_name, bias_param in [
                        ('q_proj.bias', attn.q_proj.bias),
                        ('k_proj.bias', attn.k_proj.bias),
                        ('v_proj.bias', attn.v_proj.bias),
                        ('o_proj.bias', attn.o_proj.bias),
                    ]:
                        if bias_param is not None:
                            bias_param.requires_grad = True
                            bias_trainable_count += 1
            
            if bias_trainable_count > 0:
                logger.info(f'✅ Set q_proj/k_proj/v_proj/o_proj.bias to trainable for {bias_trainable_count // 4} layers (LoRA mode, Qwen2)')
            else:
                logger.warning('⚠️  No bias parameters found to make trainable for Qwen2')

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

### mh 11/17: The notation [model_args.unfreeze_vit_layers:] means "from index 2 to the end" Layers 0-1: Frozen ✗
# Layers 2-23: Trainable ✓
    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

### mhu 11/10: use InternVLTrainer instead of Trainer
### trainer = Trainer(
####
    trainer = InternVLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # mh1122: CRITICAL: Force patch accelerator.no_sync after trainer initialization
    # This must be done after accelerator is fully initialized and before training starts
    if training_args.do_train and hasattr(trainer, 'accelerator') and trainer.is_deepspeed_enabled:
        logger.info("Patching accelerator.no_sync for DeepSpeed ZeRO Stage 2/3 compatibility...")
        trainer._patch_no_sync_for_zero()
        # Verify patch was applied
        if hasattr(trainer.accelerator, '_internvl_no_sync_patched'):
            logger.info("✅ Successfully patched accelerator.no_sync")
        else:
            logger.warning("⚠️  Patch may not have been applied correctly")

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
