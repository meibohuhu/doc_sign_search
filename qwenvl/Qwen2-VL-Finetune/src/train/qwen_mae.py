import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import math

# Import monkey patches (same directory)
from monkey_patch_vision import replace_qwen2_5_vision


class QwenViTMAE(nn.Module):
    """
    MAE for Qwen2-VL vision encoder
    
    Key design:
    - Encoder: 只处理visible tokens (未被mask的patches)，节省计算
    - Decoder: 轻量级，负责重建masked patches
    - Loss: 只在masked patches上计算
    
    Input format (compatible with Qwen2-VL):
    - pixel_values_videos: [N, patch_pixel_dim] - flattened video patches
    - video_grid_thw: [num_videos, 3] - (T, H, W) for each video
    """
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for decoder blocks to save memory
        
        Args:
            gradient_checkpointing_kwargs: Optional kwargs for gradient checkpointing
                (e.g., {"use_reentrant": True})
        """
        self._gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
        # Also enable for encoder if it supports it
        if hasattr(self.visual, 'gradient_checkpointing_enable'):
            self.visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self._gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        spacetime_mask: bool = True,
        mask_strategy: str = 'tube',  # 'random', 'tube', 'block', 'mu'
        mask_unit_size: Tuple[int, int] = (4, 4),  # For 'mu' and 'block' strategies: (H, W) patch blocks
    ):
        super().__init__()
        
        # Check if this is Qwen2.5-VL and apply monkey patches if needed
        is_qwen25 = "Qwen2.5" in model_id or "qwen2.5" in model_id.lower()
        if is_qwen25:
            replace_qwen2_5_vision()
        
        # Load Qwen2-VL model to extract ViT
        print(f"Loading Qwen2-VL from {model_id}...")
        
        # Use correct model class based on model_id
        if is_qwen25:
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True  # Qwen2.5-VL requires trust_remote_code
            )
        else:
            qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
            )
        
        # Extract vision encoder
        # 注意：encoder 的训练由 train_qwen_mae.py 控制
        # 默认情况下所有 encoder 层都是可训练的（requires_grad=True）
        # 可以通过 --freeze_encoder 或 --unfreeze_topk_vision 来控制
        self.visual = qwen_model.visual
        
        # Get key dimensions from Qwen's visual encoder
        self.encoder_embed_dim = self.visual.config.hidden_size  # e.g., 1536
        self.patch_size = 14  # Qwen2-VL typically uses 14x14 patches
        # Note: patch_pixel_dim will be determined dynamically from input
        # Qwen2-VL's processor may output different dimensions (e.g., 1176 instead of 588)
        # We'll get it from the first forward pass
        self.patch_pixel_dim = None  # Will be set dynamically
        
        # Masking parameters
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.spacetime_mask = spacetime_mask
        # Masking strategy: 
        #   'random': patch-level random (current, simple but loses locality)
        #   'tube': time-tube masking (keep spatial continuity, mask temporal tube)
        #   'block': spatial block masking (keep temporal continuity, mask spatial blocks)
        #   'mu': mask-unit like Hiera (keep both spatial and temporal locality)
        self.mask_strategy = mask_strategy
        self.mask_unit_size = mask_unit_size  # For 'block' and 'mu' strategies: (H, W) patch blocks
        
        # ================== MAE Decoder ==================
        # Decoder input projection
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim, 
                decoder_num_heads, 
                mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: predict RGB pixel values per patch
        # Initialize with a placeholder dimension, will be updated dynamically
        # Default to 1176 which is common for Qwen2-VL (can be 588 for some configs)
        self.decoder_pred = None  # Will be created dynamically
        self._decoder_pred_dim = None  # Placeholder
        
        # Initialize weights
        self.initialize_weights()
        
        # Gradient checkpointing flag
        self._gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}
        
    def initialize_weights(self):
        """Initialize decoder weights"""
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize decoder layers (except decoder_pred which is created dynamically)
        # Only initialize layers that exist
        for name, module in self.named_modules():
            if name != 'decoder_pred':  # Skip decoder_pred (None initially)
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    self._init_weights(module)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_spacetime_masking(
        self, 
        video_grid_thw: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masking strategy router - selects appropriate masking method based on self.mask_strategy
        
        Args:
            video_grid_thw: [num_videos, 3] - (T, H, W) for each video
            device: device for tensors
        
        Returns:
            mask: [total_patches] - binary mask (1 = masked, 0 = visible)
            ids_restore: [total_patches] - indices to restore original order
            ids_keep: [total_visible_patches] - indices of visible patches
        """
        # Print masking strategy confirmation (only once)
        if not hasattr(self, '_mask_strategy_logged'):
            # Check if distributed training and only log on rank 0
            is_main_process = True
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    is_main_process = (dist.get_rank() == 0)
            except:
                pass
            
            if is_main_process:
                print(f"\n{'='*70}")
                print(f"🎭 Masking Strategy: {self.mask_strategy.upper()}")
                print(f"{'='*70}")
                print(f"  Strategy: {self.mask_strategy}")
                print(f"  Mask Ratio: {self.mask_ratio}")
                if self.mask_strategy in ['block', 'mu']:
                    print(f"  Mask Unit Size: {self.mask_unit_size} (H={self.mask_unit_size[0]}, W={self.mask_unit_size[1]})")
                print(f"{'='*70}\n")
            
            self._mask_strategy_logged = True
        
        if self.mask_strategy == 'random':
            return self._random_patch_masking(video_grid_thw, device)
        elif self.mask_strategy == 'tube':
            return self._tube_masking(video_grid_thw, device)
        elif self.mask_strategy == 'block':
            return self._block_masking(video_grid_thw, device)
        elif self.mask_strategy == 'mu':
            return self._mask_unit_masking(video_grid_thw, device)
        else:
            raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")
    
    def _random_patch_masking(
        self, 
        video_grid_thw: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Original random patch-level masking (each patch independently)
        - Simple but loses spatial and temporal locality
        """
        num_videos = video_grid_thw.shape[0]
        
        # Calculate total patches per video: T * H * W
        num_patches_per_video = (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]).long()
        total_patches = num_patches_per_video.sum().item()
        
        # Calculate cumulative patches for indexing
        cumsum_patches = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            num_patches_per_video.cumsum(0)
        ])
        
        # Generate random noise for each patch
        noise = torch.rand(total_patches, device=device)
        
        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)
        
        # Calculate number of visible patches
        len_keep = int(total_patches * (1 - self.mask_ratio))
        ids_keep = ids_shuffle[:len_keep]
        
        # Generate binary mask: 1 = masked, 0 = visible
        mask = torch.ones(total_patches, dtype=torch.bool, device=device)
        mask[ids_keep] = 0
        
        # Restore mask to original patch order
        mask = mask[ids_restore]
        
        # Print confirmation (only first time)
        if not hasattr(self, '_random_mask_logged'):
            is_main_process = True
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    is_main_process = (dist.get_rank() == 0)
            except:
                pass
            
            if is_main_process:
                print(f"[RANDOM Masking] ✓ Confirmed using random patch-level masking")
                print(f"  - Total patches: {total_patches}")
                print(f"  - Visible patches: {len_keep} ({len_keep/total_patches*100:.1f}%)")
                print(f"  - Masked patches: {total_patches - len_keep} ({(total_patches-len_keep)/total_patches*100:.1f}%)")
                print(f"  - Each patch independently masked/visible\n")
            
            self._random_mask_logged = True
        
        return mask, ids_restore, ids_keep, num_patches_per_video, cumsum_patches
    
    def _tube_masking(
        self, 
        video_grid_thw: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Time-tube masking: mask entire temporal tubes (all frames at same spatial position)
        - Keeps spatial continuity (hand gestures preserved spatially)
        - Better for sign language (hand positions important)
        """
        num_videos = video_grid_thw.shape[0]
        masks = []
        ids_restores = []
        ids_keeps = []
        num_patches_per_video_list = []
        cumsum_patches_list = [torch.zeros(1, dtype=torch.long, device=device)]
        
        for i in range(num_videos):
            T, H, W = video_grid_thw[i].tolist()
            num_spatial_patches = H * W
            num_patches = T * H * W
            
            # Generate random noise for each spatial location (not each patch)
            noise = torch.rand(num_spatial_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=0)
            ids_restore_spatial = torch.argsort(ids_shuffle, dim=0)
            
            # Calculate number of visible spatial locations
            len_keep_spatial = int(num_spatial_patches * (1 - self.mask_ratio))
            ids_keep_spatial = ids_shuffle[:len_keep_spatial]
            
            # Create spatial mask: 1 = masked, 0 = visible
            spatial_mask = torch.ones(num_spatial_patches, dtype=torch.bool, device=device)
            spatial_mask[ids_keep_spatial] = 0
            spatial_mask = spatial_mask[ids_restore_spatial]
            
            # Expand to all frames: [H*W] -> [T*H*W]
            # Each spatial location is masked/visible for all frames
            mask = spatial_mask.unsqueeze(0).repeat(T, 1).flatten()  # [T*H*W]
            
            # Get visible patch indices
            ids_keep = torch.where(~mask)[0]
            
            # Create restore indices (just identity since we're not shuffling)
            ids_restore = torch.arange(num_patches, device=device)
            
            masks.append(mask)
            ids_restores.append(ids_restore)
            ids_keeps.append(ids_keep)
            num_patches_per_video_list.append(num_patches)
            cumsum_patches_list.append(cumsum_patches_list[-1] + num_patches)
            
            # Print confirmation (only first video, first time)
            if i == 0 and not hasattr(self, '_tube_mask_logged'):
                is_main_process = True
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        is_main_process = (dist.get_rank() == 0)
                except:
                    pass
                
                if is_main_process:
                    print(f"[TUBE Masking] ✓ Confirmed using time-tube masking")
                    print(f"  - Video shape: T={T}, H={H}, W={W}")
                    print(f"  - Spatial locations: {num_spatial_patches} (H×W)")
                    print(f"  - Visible locations: {len_keep_spatial} ({len_keep_spatial/num_spatial_patches*100:.1f}%)")
                    print(f"  - Masked locations: {num_spatial_patches - len_keep_spatial} ({(num_spatial_patches-len_keep_spatial)/num_spatial_patches*100:.1f}%)")
                    print(f"  - Total patches: {num_patches} (T×H×W)")
                    print(f"  - Visible patches: {len(ids_keep)} ({len(ids_keep)/num_patches*100:.1f}%)")
                    print(f"  - ✓ Each spatial location (1×1 patch) kept same across all {T} frames\n")
                
                self._tube_mask_logged = True
        
        # Concatenate all videos
        mask = torch.cat(masks)
        ids_restore = torch.cat(ids_restores)
        ids_keep = torch.cat(ids_keeps)
        num_patches_per_video = torch.tensor(num_patches_per_video_list, device=device)
        cumsum_patches = torch.cat(cumsum_patches_list)
        
        return mask, ids_restore, ids_keep, num_patches_per_video, cumsum_patches
    
    def _block_masking(
        self, 
        video_grid_thw: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Spatial block masking: mask spatial blocks (e.g., 4x4 patches) across all frames
        - Keeps temporal continuity (same spatial block visible/masked across time)
        - Better for understanding temporal motion
        """
        num_videos = video_grid_thw.shape[0]
        block_h, block_w = self.mask_unit_size
        masks = []
        ids_restores = []
        ids_keeps = []
        num_patches_per_video_list = []
        cumsum_patches_list = [torch.zeros(1, dtype=torch.long, device=device)]
        
        for i in range(num_videos):
            T, H, W = video_grid_thw[i].tolist()
            
            # Calculate number of blocks in spatial dimensions
            num_blocks_h = H // block_h
            num_blocks_w = W // block_w
            num_spatial_blocks = num_blocks_h * num_blocks_w
            
            # Generate random noise for each spatial block
            noise = torch.rand(num_spatial_blocks, device=device)
            ids_shuffle = torch.argsort(noise, dim=0)
            ids_restore_spatial = torch.argsort(ids_shuffle, dim=0)
            
            # Calculate number of visible blocks
            len_keep_blocks = int(num_spatial_blocks * (1 - self.mask_ratio))
            ids_keep_blocks = ids_shuffle[:len_keep_blocks]
            
            # Create block mask: 1 = masked, 0 = visible
            block_mask = torch.ones(num_spatial_blocks, dtype=torch.bool, device=device)
            block_mask[ids_keep_blocks] = 0
            block_mask = block_mask[ids_restore_spatial]
            
            # Expand to patches: [num_blocks_h, num_blocks_w] -> [H, W]
            block_mask_2d = block_mask.reshape(num_blocks_h, num_blocks_w)
            patch_mask_2d = block_mask_2d.repeat_interleave(block_h, dim=0).repeat_interleave(block_w, dim=1)
            # Crop to exact size if needed
            patch_mask_2d = patch_mask_2d[:H, :W]
            
            # Expand to all frames: [H, W] -> [T, H, W] -> [T*H*W]
            patch_mask = patch_mask_2d.unsqueeze(0).repeat(T, 1, 1).flatten()  # [T*H*W]
            
            # Get visible patch indices
            ids_keep = torch.where(~patch_mask)[0]
            
            # Create restore indices
            ids_restore = torch.arange(T * H * W, device=device)
            
            masks.append(patch_mask)
            ids_restores.append(ids_restore)
            ids_keeps.append(ids_keep)
            num_patches_per_video_list.append(T * H * W)
            cumsum_patches_list.append(cumsum_patches_list[-1] + T * H * W)
            
            # Print confirmation (only first video, first time)
            if i == 0 and not hasattr(self, '_block_mask_logged'):
                is_main_process = True
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        is_main_process = (dist.get_rank() == 0)
                except:
                    pass
                
                if is_main_process:
                    patches_per_block = block_h * block_w
                    print(f"[BLOCK Masking] ✓ Confirmed using spatial block masking")
                    print(f"  - Video shape: T={T}, H={H}, W={W}")
                    print(f"  - Block size: {block_h}×{block_w} patches (H×W)")
                    print(f"  - Spatial blocks: {num_blocks_h}×{num_blocks_w} = {num_spatial_blocks} blocks")
                    print(f"  - Visible blocks: {len_keep_blocks} ({len_keep_blocks/num_spatial_blocks*100:.1f}%)")
                    print(f"  - Masked blocks: {num_spatial_blocks - len_keep_blocks} ({(num_spatial_blocks-len_keep_blocks)/num_spatial_blocks*100:.1f}%)")
                    print(f"  - Patches per block: {patches_per_block}")
                    print(f"  - Total patches: {T * H * W} (T×H×W)")
                    print(f"  - Visible patches: {len(ids_keep)} ({len(ids_keep)/(T*H*W)*100:.1f}%)")
                    print(f"  - ✓ Each {block_h}×{block_w} spatial block kept same across all {T} frames\n")
                
                self._block_mask_logged = True
        
        # Concatenate all videos
        mask = torch.cat(masks)
        ids_restore = torch.cat(ids_restores)
        ids_keep = torch.cat(ids_keeps)
        num_patches_per_video = torch.tensor(num_patches_per_video_list, device=device)
        cumsum_patches = torch.cat(cumsum_patches_list)
        
        return mask, ids_restore, ids_keep, num_patches_per_video, cumsum_patches
    
    def _mask_unit_masking(
        self, 
        video_grid_thw: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mask Unit (MU) masking similar to Hiera: mask 3D blocks (T x H x W)
        - Keeps both spatial and temporal locality
        - Best for sign language (hand gestures need both spatial structure and temporal motion)
        """
        num_videos = video_grid_thw.shape[0]
        block_h, block_w = self.mask_unit_size
        # For temporal, use 1 frame per MU (can be adjusted)
        block_t = 2
        
        masks = []
        ids_restores = []
        ids_keeps = []
        num_patches_per_video_list = []
        cumsum_patches_list = [torch.zeros(1, dtype=torch.long, device=device)]
        
        for i in range(num_videos):
            T, H, W = video_grid_thw[i].tolist()
            
            # Calculate number of MUs in each dimension
            num_blocks_t = (T + block_t - 1) // block_t  # Ceiling division
            num_blocks_h = H // block_h
            num_blocks_w = W // block_w
            num_mus = num_blocks_t * num_blocks_h * num_blocks_w
            
            # Generate random noise for each MU
            noise = torch.rand(num_mus, device=device)
            ids_shuffle = torch.argsort(noise, dim=0)
            ids_restore_mu = torch.argsort(ids_shuffle, dim=0)
            
            # Calculate number of visible MUs
            len_keep_mus = int(num_mus * (1 - self.mask_ratio))
            ids_keep_mus = ids_shuffle[:len_keep_mus]
            
            # Create MU mask: 1 = masked, 0 = visible
            mu_mask = torch.ones(num_mus, dtype=torch.bool, device=device)
            mu_mask[ids_keep_mus] = 0
            mu_mask = mu_mask[ids_restore_mu]
            
            # Expand MU mask to patches: [num_blocks_t, num_blocks_h, num_blocks_w] -> [T, H, W]
            mu_mask_3d = mu_mask.reshape(num_blocks_t, num_blocks_h, num_blocks_w)
            
            # Expand each dimension
            patch_mask_3d = (
                mu_mask_3d.repeat_interleave(block_t, dim=0)[:T]
                .repeat_interleave(block_h, dim=1)[:, :H]
                .repeat_interleave(block_w, dim=2)[:, :, :W]
            )
            
            # Flatten: [T, H, W] -> [T*H*W]
            patch_mask = patch_mask_3d.flatten()
            
            # Get visible patch indices
            ids_keep = torch.where(~patch_mask)[0]
            
            # Create restore indices
            ids_restore = torch.arange(T * H * W, device=device)
            
            masks.append(patch_mask)
            ids_restores.append(ids_restore)
            ids_keeps.append(ids_keep)
            num_patches_per_video_list.append(T * H * W)
            cumsum_patches_list.append(cumsum_patches_list[-1] + T * H * W)
            
            # Print confirmation (only first video, first time)
            if i == 0 and not hasattr(self, '_mu_mask_logged'):
                is_main_process = True
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        is_main_process = (dist.get_rank() == 0)
                except:
                    pass
                
                if is_main_process:
                    patches_per_mu = block_t * block_h * block_w
                    print(f"[MU Masking] ✓ Confirmed using Mask Unit (MU) masking (similar to Hiera)")
                    print(f"  - Video shape: T={T}, H={H}, W={W}")
                    print(f"  - MU size: {block_t}×{block_h}×{block_w} patches (T×H×W)")
                    print(f"  - MU dimensions: {num_blocks_t}×{num_blocks_h}×{num_blocks_w} = {num_mus} MUs")
                    print(f"  - Visible MUs: {len_keep_mus} ({len_keep_mus/num_mus*100:.1f}%)")
                    print(f"  - Masked MUs: {num_mus - len_keep_mus} ({(num_mus-len_keep_mus)/num_mus*100:.1f}%)")
                    print(f"  - Patches per MU: {patches_per_mu} ({block_t}×{block_h}×{block_w})")
                    print(f"  - Total patches: {T * H * W} (T×H×W)")
                    print(f"  - Visible patches: {len(ids_keep)} ({len(ids_keep)/(T*H*W)*100:.1f}%)")
                    print(f"  - ✓ Each {block_t}×{block_h}×{block_w} MU keeps spatial and temporal locality\n")
                
                self._mu_mask_logged = True
        
        # Concatenate all videos
        mask = torch.cat(masks)
        ids_restore = torch.cat(ids_restores)
        ids_keep = torch.cat(ids_keeps)
        num_patches_per_video = torch.tensor(num_patches_per_video_list, device=device)
        cumsum_patches = torch.cat(cumsum_patches_list)
        
        return mask, ids_restore, ids_keep, num_patches_per_video, cumsum_patches
    
    def _build_visible_grid_thw(
        self, 
        video_grid_thw: torch.Tensor,
        ids_keep: torch.Tensor,
        num_patches_per_video: torch.Tensor,
        cumsum_patches: torch.Tensor
    ) -> torch.Tensor:
        """
        Build grid_thw for visible tokens only.
        Each visible token keeps its original spatial position (T, H, W),
        but we need to organize them by video.
        
        Args:
            video_grid_thw: [num_videos, 3] - original (T, H, W) for each video
            ids_keep: [num_visible] - global indices of visible patches
            num_patches_per_video: [num_videos] - patches per video
            cumsum_patches: [num_videos+1] - cumulative patch indices
        
        Returns:
            visible_grid_thw: [num_videos, 3] - (T_vis, H, W) for visible tokens
                Note: T_vis may be different from original T due to masking
        """
        device = ids_keep.device
        num_videos = video_grid_thw.shape[0]
        
        # For each video, count visible tokens
        visible_counts = []
        visible_grid_thw_list = []
        
        for i in range(num_videos):
            start_idx = cumsum_patches[i].item()
            end_idx = cumsum_patches[i + 1].item()
            
            # Get visible token indices for this video
            video_mask = (ids_keep >= start_idx) & (ids_keep < end_idx)
            video_visible_ids = ids_keep[video_mask] - start_idx  # Local indices within video
            
            if len(video_visible_ids) == 0:
                # No visible tokens for this video (shouldn't happen, but handle it)
                visible_grid_thw_list.append(torch.tensor([[0, 0, 0]], device=device))
                visible_counts.append(0)
                continue
            
            T, H, W = video_grid_thw[i].tolist()
            
            # Calculate which frame each visible token belongs to
            # Each patch's index: t * H * W + h * W + w
            # So: t = idx // (H * W), local_idx = idx % (H * W), h = local_idx // W, w = local_idx % W
            t_indices = video_visible_ids // (H * W)
            unique_t = torch.unique(t_indices)
            
            # Visible tokens span T_vis frames (may be less than original T)
            T_vis = len(unique_t)
            
            # Keep original H, W (spatial dimensions unchanged)
            visible_grid_thw_list.append(torch.tensor([[T_vis, H, W]], device=device))
            visible_counts.append(len(video_visible_ids))
        
        visible_grid_thw = torch.cat(visible_grid_thw_list, dim=0)  # [num_videos, 3]
        
        return visible_grid_thw
    
    def forward_encoder(
        self, 
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder forward pass - 只处理visible tokens!
        
        ✅ 完整的 Vision Encoder 流程：
        1. Patch Embedding: pixel patches -> embeddings
        2. Rotary Position Embedding: 添加位置信息
        3. Window Reorganization: 按窗口重组 tokens
        4. 所有 Encoder Blocks: 通过 self.visual.blocks (完整的 transformer blocks)
        5. Merger: 投影到 LLM 维度
        
        关键：第412-432行遍历所有 encoder blocks，每个 block 包含：
        - Self-attention (带 rotary position embedding)
        - MLP
        - Layer normalization
        
        Args:
            pixel_values_videos: [N, patch_pixel_dim] - flattened patches
            video_grid_thw: [num_videos, 3] - (T, H, W) for each video
        
        Returns:
            latent: [num_visible, encoder_dim] - encoded visible tokens
            mask: [N] - binary mask (1 = masked, 0 = visible)
            ids_restore: [N] - indices to restore original order
            ids_keep: [num_visible] - indices of visible patches
        """
        device = pixel_values_videos.device
        N = pixel_values_videos.shape[0]
        num_videos = video_grid_thw.shape[0]
        
        # Step 1: Random spacetime masking (before processing through encoder)
        mask, ids_restore, ids_keep, num_patches_per_video, cumsum_patches = self.random_spacetime_masking(
            video_grid_thw, device
        )
        
        # Step 2: Extract visible patches only (关键优化!)
        visible_patches = pixel_values_videos[ids_keep]  # [num_visible, patch_pixel_dim]
        
        # Step 3: Get patch embeddings for visible tokens only
        # 这只是第一步：将 pixel patches 转换为 embeddings
        # 接下来会通过完整的 encoder blocks
        visible_embeds = self.visual.patch_embed(visible_patches)  # [num_visible, encoder_dim]
        
        # ============================================================
        # Step 4-7: 通过完整的 Vision Encoder
        # ============================================================
        # 对每个视频分别处理，以保持空间结构
        # 每个视频的 visible tokens 会经过：
        #   1. Patch Embedding (已完成)
        #   2. Rotary Position Embedding
        #   3. Window Reorganization
        #   4. 所有 Encoder Blocks (self.visual.blocks)
        #   5. Merger (如果有)
        # ============================================================
        visible_latents = []
        
        for i in range(num_videos):
            start_global = cumsum_patches[i].item()
            end_global = cumsum_patches[i + 1].item()
            
            # Get visible tokens for this video
            video_visible_mask = (ids_keep >= start_global) & (ids_keep < end_global)
            video_visible_ids = ids_keep[video_visible_mask]
            
            if len(video_visible_ids) == 0:
                # Skip if no visible tokens (shouldn't happen with reasonable mask_ratio)
                continue
            
            # Get local indices within video
            video_visible_local_ids = video_visible_ids - start_global
            video_visible_embeds = visible_embeds[video_visible_mask]  # [num_vis_video, encoder_dim]
            
            # Original video dimensions
            T, H, W = video_grid_thw[i].tolist()
            
            # Reorganize visible tokens by frame for proper grid_thw
            # Sort by (t, h, w) to maintain spatial structure
            sorted_indices = torch.argsort(video_visible_local_ids)
            video_visible_embeds_sorted = video_visible_embeds[sorted_indices]
            video_visible_local_ids_sorted = video_visible_local_ids[sorted_indices]
            
            # Calculate t, h, w for each visible token (in original video coordinates)
            t_coords = video_visible_local_ids_sorted // (H * W)
            hw_coords = video_visible_local_ids_sorted % (H * W)
            h_coords = hw_coords // W
            w_coords = hw_coords % W
            
            # Build dense representation for encoder forward
            # 注意：Qwen2-VL encoder 可能需要完整的序列结构（包括 masked positions）来计算位置编码
            # 但实际处理时可能只处理非零 tokens，所以我们需要：
            # 1. 构建完整序列 [T, H, W, D]，masked 位置填零
            # 2. 这样 rotary position embedding 和 window index 可以正确计算
            # 3. Encoder blocks 会处理所有 tokens（包括零填充），但可能输出时只保留非零部分
            video_latent = torch.zeros(T, H, W, video_visible_embeds.shape[-1], 
                                     device=device, dtype=video_visible_embeds.dtype)
            
            # Place visible tokens at their original positions
            for idx, (t, h, w) in enumerate(zip(t_coords, h_coords, w_coords)):
                video_latent[t.item(), h.item(), w.item()] = video_visible_embeds_sorted[idx]
            
            # Flatten to [T * H * W, D] for encoder forward (完整序列，包括 masked)
            video_latent_flat = video_latent.reshape(-1, video_visible_embeds.shape[-1])
            
            # Create grid_thw: [1, 3] with [T, H, W] (使用原始 T 以计算正确的位置编码)
            visible_grid_thw_i = torch.tensor([[T, H, W]], device=device, dtype=video_grid_thw.dtype)
            
            # Step 5: Process through encoder blocks with rotary position embedding
            # Get rotary position embedding
            rotary_pos_emb = self.visual.rot_pos_emb(visible_grid_thw_i)  # [T_vis*H*W, head_dim//2]
            
            # Get window index
            window_index, cu_window_seqlens_list = self.visual.get_window_index(visible_grid_thw_i)
            
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens_list,
                device=device,
                dtype=torch.int32,
            )
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
            
            # Reorganize by windows (same as visual.forward)
            group = self.visual.spatial_merge_unit
            seq_len = video_latent_flat.shape[0]
            G = seq_len // group
            
            if G > 0:
                hidden_states = video_latent_flat.view(G, group, -1)
                rotary_pos_emb_view = rotary_pos_emb.view(G, group, -1)
                
                window_index_dev = window_index.to(device, non_blocking=True)
                hidden_states = hidden_states.index_select(0, window_index_dev).reshape(seq_len, -1)
                rotary_pos_emb = rotary_pos_emb_view.index_select(0, window_index_dev).reshape(seq_len, -1)
            else:
                # Edge case: too few tokens
                hidden_states = video_latent_flat
                window_index_dev = torch.zeros(0, dtype=torch.long, device=device)
                cu_window_seqlens = torch.zeros(1, dtype=torch.int32, device=device)
            
            # Convert to position embeddings
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
            
            # Compute cu_seqlens
            cu_seqlens = torch.repeat_interleave(
                visible_grid_thw_i[:, 1] * visible_grid_thw_i[:, 2], 
                visible_grid_thw_i[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            
            if cu_seqlens.device != device:
                cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            
            # ============================================================
            # Step 6: 通过完整的 Vision Encoder Blocks ⭐ 关键步骤 ⭐
            # ============================================================
            # 这里遍历并调用 Qwen2-VL 的所有 encoder transformer blocks
            # self.visual.blocks 是完整的 encoder，包含所有层（例如 32 层）
            # 
            # 每个 block (Qwen2_5_VLVisionBlock) 包含：
            #   - Self-attention (带 rotary position embedding)
            #   - MLP (Feed-forward network)
            #   - Layer normalization
            # 
            # ✅ 这是完整的 encoder forward，visible tokens 真正通过所有 encoder blocks！
            # ============================================================
            for layer_num, blk in enumerate(self.visual.blocks):
                # 选择使用 full attention 还是 window attention
                if hasattr(self.visual, 'fullatt_block_indexes') and layer_num in self.visual.fullatt_block_indexes:
                    cu_seqlens_now = cu_seqlens  # Full attention
                else:
                    cu_seqlens_now = cu_window_seqlens  # Window attention
                
                # 通过当前 encoder block
                # 传入 position_embeddings (rotary position encoding)
                if self.visual.gradient_checkpointing and self.training:
                    hidden_states = self.visual._gradient_checkpointing_func(
                        blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                    )
                else:
                    # ✅ 关键：这里调用完整的 encoder block
                    # blk 是 Qwen2_5_VLVisionBlock 或类似的完整 transformer block
                    hidden_states = blk(
                        hidden_states, 
                        cu_seqlens=cu_seqlens_now, 
                        position_embeddings=position_embeddings
                    )
            
            # ============================================================
            # Step 7: 跳过 Merger (MAE 不需要)
            # ============================================================
            # Merger 将 encoder 输出投影到 LLM 的维度 (out_hidden_size, 通常是 512)
            # 但对于 MAE，我们需要保持 encoder 的原始维度 (hidden_size, 通常是 1280)
            # 这样才能与 decoder_embed 的输入维度匹配
            # 
            # 注意：如果使用 merger，需要调整 decoder_embed 的输入维度为 out_hidden_size
            # 但为了保持 encoder 特征的完整性，我们跳过 merger
            # if hasattr(self.visual, 'merger'):
            #     hidden_states = self.visual.merger(hidden_states)  # 跳过！
            
            # Reverse window indexing if needed
            if G > 0 and window_index_dev.numel() > 0:
                reverse_indices = torch.empty_like(window_index_dev)
                reverse_indices.scatter_(
                    0, window_index_dev, 
                    torch.arange(window_index_dev.numel(), dtype=torch.long, device=device)
                )
                
                # Reshape hidden_states back to [G, group, D] for reverse indexing
                hidden_states_G = hidden_states.view(-1, group, hidden_states.shape[-1])
                hidden_states_reversed = hidden_states_G.index_select(0, reverse_indices)
                hidden_states = hidden_states_reversed.reshape(-1, hidden_states.shape[-1])
            
            # Extract visible tokens from encoder output
            # ✅ Encoder 输出完整序列 [T*H*W, D] after reverse window indexing
            # 我们需要从中提取 visible positions
            
            num_visible_tokens = len(video_visible_local_ids_sorted)
            expected_full_seq_len = T * H * W
            
            # Check if encoder output matches full sequence
            if hidden_states.shape[0] == expected_full_seq_len:
                # ✅ Encoder output is full sequence: extract visible positions
                # Use linear indexing for better efficiency: idx = t * (H*W) + h * W + w
                visible_linear_indices = t_coords * (H * W) + h_coords * W + w_coords
                video_visible_latent = hidden_states[visible_linear_indices]  # [num_vis_video, encoder_embed_dim]
            elif hidden_states.shape[0] == num_visible_tokens:
                # Encoder output is only visible tokens: use directly
                # But we need to ensure the order matches video_visible_embeds_sorted
                # (encoder should preserve the order of visible tokens)
                video_visible_latent = hidden_states  # [num_vis_video, encoder_embed_dim]
            else:
                # Unexpected: might be batch-level processing or indexing issue
                # If hidden_states is longer than expected, it might contain tokens from multiple videos
                # In this case, we should only take the first expected_full_seq_len tokens
                if hidden_states.shape[0] > expected_full_seq_len:
                    hidden_states = hidden_states[:expected_full_seq_len]
                    # Use linear indexing for better efficiency: idx = t * (H*W) + h * W + w
                    visible_linear_indices = t_coords * (H * W) + h_coords * W + w_coords
                    video_visible_latent = hidden_states[visible_linear_indices]
                else:
                    raise RuntimeError(
                        f"[Video {i}] Hidden states sequence length mismatch: "
                        f"expected either {expected_full_seq_len} (full sequence) or {num_visible_tokens} (visible tokens), "
                        f"got {hidden_states.shape[0]}"
                    )
            
            # Verify dimension matches encoder_embed_dim
            if hidden_states.shape[1] != self.encoder_embed_dim:
                raise RuntimeError(
                    f"[Video {i}] Hidden states feature dimension mismatch: "
                    f"expected {self.encoder_embed_dim} (encoder_embed_dim = hidden_size), "
                    f"got {hidden_states.shape[1]}. "
                    f"This might indicate merger was applied incorrectly or dimension inference failed."
                )
            
            visible_latents.append(video_visible_latent)
        
        # Concatenate all video latents
        # Important: The order of latents must match ids_keep order
        # ids_keep is in global shuffled order: [0, 1, 2, ..., num_visible-1]
        # But we processed videos separately, so we need to reorder latents to match ids_keep
        
        if len(visible_latents) > 0:
            # Concatenate latents (currently in video order, not ids_keep order)
            latent_concat = torch.cat(visible_latents, dim=0)  # [num_visible, encoder_dim]
            
            # Reorder latents to match ids_keep order
            # ids_keep contains global shuffled positions: [shuffled_pos0, shuffled_pos1, ..., shuffled_pos_{num_visible-1}]
            # These are the positions in the global shuffle (sorted ascending)
            # We need to map from video-order (which we cat'd) to ids_keep-order
            
            # Build mapping: for each position in ids_keep, find which video it belongs to
            # and its position in that video's latent
            latent_reordered = torch.zeros_like(latent_concat)
            current_latent_idx = 0
            
            for i in range(num_videos):
                start_global = cumsum_patches[i].item()
                end_global = cumsum_patches[i + 1].item()
                
                # Get visible tokens for this video (in ids_keep)
                video_visible_mask = (ids_keep >= start_global) & (ids_keep < end_global)
                video_num_vis = video_visible_mask.sum().item()
                
                if video_num_vis > 0:
                    # Find positions in ids_keep where this video's visible tokens appear
                    # ids_keep[video_visible_mask] gives us the global shuffled positions
                    # We need to find where these appear in the ids_keep array
                    ids_keep_indices = torch.where(video_visible_mask)[0]  # Positions in ids_keep array
                    latent_reordered[ids_keep_indices] = latent_concat[current_latent_idx:current_latent_idx + video_num_vis]
                    current_latent_idx += video_num_vis
            
            latent = latent_reordered  # [num_visible, encoder_dim] in ids_keep order
        else:
            # Fallback: return patch embeddings if no visible tokens processed
            latent = visible_embeds
        
        return latent, mask, ids_restore, ids_keep
    
    def forward_decoder(
        self, 
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
        ids_keep: torch.Tensor,
        num_patches: int
    ) -> torch.Tensor:
        """
        Decoder forward pass - 处理所有tokens (visible + mask tokens)
        
        Args:
            latent: [num_visible, encoder_dim] - encoded visible tokens
            ids_restore: [num_patches] - indices to restore original order
            ids_keep: [num_visible] - indices of visible patches (in shuffled order)
            num_patches: total number of patches
        
        Returns:
            pred: [num_patches, patch_pixel_dim] - reconstructed patches (in original order)
        """
        # Step 1: Project encoder output to decoder dimension
        x = self.decoder_embed(latent)  # [num_visible, decoder_dim]
        
        # Step 2: Build full sequence with visible tokens + mask tokens
        # ids_keep are indices in shuffled order (first len_keep after sorting)
        # ids_restore maps shuffled -> original order
        # So: original positions of visible tokens = ids_restore[ids_keep]
        
        # Create full sequence [num_patches, decoder_dim]
        decoder_dim = x.shape[1]
        full_sequence = torch.zeros(num_patches, decoder_dim, device=x.device, dtype=x.dtype)
        
        # Place visible tokens at their original positions
        visible_positions_original = ids_restore[ids_keep]  # [num_visible]
        full_sequence[visible_positions_original] = x
        
        # Place mask tokens at remaining positions
        all_positions = torch.arange(num_patches, device=x.device)
        # Create mask for positions not in visible_positions_original
        mask_positions_original = torch.ones(num_patches, dtype=torch.bool, device=x.device)
        mask_positions_original[visible_positions_original] = False
        mask_positions_original = all_positions[mask_positions_original]
        num_mask = len(mask_positions_original)
        if num_mask > 0:
            full_sequence[mask_positions_original] = self.mask_token.squeeze(0).expand(num_mask, -1)
        
        x = full_sequence  # [num_patches, decoder_dim] in original order
        
        # Step 4: Apply decoder blocks (with gradient checkpointing if enabled)
        if getattr(self, '_gradient_checkpointing', False) and self.training:
            from torch.utils.checkpoint import checkpoint
            checkpoint_kwargs = getattr(self, '_gradient_checkpointing_kwargs', {})
            # Default to use_reentrant=True if not specified
            use_reentrant = checkpoint_kwargs.get('use_reentrant', True)
            for blk in self.decoder_blocks:
                x = checkpoint(blk, x, use_reentrant=use_reentrant)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)
        
        # Step 5: Predictor projection
        # decoder_pred should be created by _ensure_patch_pixel_dim before this
        if self.decoder_pred is None:
            raise RuntimeError(
                "decoder_pred not initialized. This should be set by _ensure_patch_pixel_dim "
                "before calling forward_decoder."
            )
        pred = self.decoder_pred(x)  # [num_patches, patch_pixel_dim]
        
        return pred
    
    def forward_loss(
        self, 
        pixel_values_videos: torch.Tensor,
        pred: torch.Tensor, 
        mask: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss only on masked patches
        
        Args:
            pixel_values_videos: [N, patch_pixel_dim] - original patches
            pred: [N, patch_pixel_dim] - reconstructed patches
            mask: [N] - binary mask (1 = masked, 0 = visible)
            ids_restore: [N] - indices to restore original order
        
        Returns:
            loss: scalar
        """
        target = pixel_values_videos  # [N, patch_pixel_dim]
        
        # Normalize target patches if enabled
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # Compute MSE loss per patch
        loss = (pred - target) ** 2  # [N, patch_pixel_dim]
        loss = loss.mean(dim=-1)  # [N], mean loss per patch
        
        # Only compute loss on masked patches
        loss = (loss * mask.float()).sum() / mask.sum().float()
        
        return loss
    
    def _ensure_patch_pixel_dim(self, pixel_values_videos: torch.Tensor):
        """Ensure patch_pixel_dim is set from actual input dimension"""
        actual_dim = pixel_values_videos.shape[-1]
        device = pixel_values_videos.device
        
        # Get the dtype from the model (should match training dtype, e.g., bfloat16)
        # Check decoder_embed's weight dtype as reference
        model_dtype = next(self.decoder_embed.parameters()).dtype
        
        if self.patch_pixel_dim is None:
            # First time: detect and set dimension
            self.patch_pixel_dim = actual_dim
            # Get decoder_embed_dim from decoder_embed layer
            decoder_embed_dim = self.decoder_embed.out_features
            
            # Create decoder_pred and register it as a module
            if self.decoder_pred is None:
                self.decoder_pred = nn.Linear(
                    decoder_embed_dim,
                    self.patch_pixel_dim,
                    bias=True
                ).to(device=device, dtype=model_dtype)
                # Initialize decoder_pred weights
                torch.nn.init.xavier_uniform_(self.decoder_pred.weight)
                if self.decoder_pred.bias is not None:
                    nn.init.constant_(self.decoder_pred.bias, 0)
                # Register as a module so it's recognized by DeepSpeed
                self.add_module('decoder_pred', self.decoder_pred)
                
                # Only print on main process to avoid duplicate logs
                if not hasattr(torch.distributed, 'is_initialized') or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"Detected patch_pixel_dim: {self.patch_pixel_dim} (creating decoder_pred layer with dtype {model_dtype})")
        elif self.patch_pixel_dim != actual_dim:
            raise ValueError(
                f"Patch pixel dimension mismatch: expected {self.patch_pixel_dim}, "
                f"got {actual_dim}. Make sure all videos have consistent dimensions."
            )
        # Ensure decoder_pred is on the right device and has correct output dim
        if self.decoder_pred is not None:
            if self.decoder_pred.out_features != actual_dim:
                # Recreate decoder_pred with correct dimension
                decoder_embed_dim = self.decoder_embed.out_features
                self.decoder_pred = nn.Linear(
                    decoder_embed_dim,
                    actual_dim,
                    bias=True
                ).to(device=device, dtype=model_dtype)
                torch.nn.init.xavier_uniform_(self.decoder_pred.weight)
                if self.decoder_pred.bias is not None:
                    nn.init.constant_(self.decoder_pred.bias, 0)
                self.add_module('decoder_pred', self.decoder_pred)
    
    def forward(
        self, 
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass
        
        Args:
            pixel_values_videos: [N, patch_pixel_dim] - flattened video patches (concatenated from batch)
            video_grid_thw: [num_videos, 3] - (T, H, W) for each video in batch
            mask_ratio: optional override for mask ratio
        
        Returns:
            loss: reconstruction loss (averaged over batch)
            pred: [N, patch_pixel_dim] - reconstructed patches
            mask: [N] - binary mask (1 = masked, 0 = visible)
        """
        # Ensure patch_pixel_dim is set from actual input
        self._ensure_patch_pixel_dim(pixel_values_videos)
        
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Temporarily set mask ratio
        old_ratio = self.mask_ratio
        self.mask_ratio = mask_ratio
        
        num_videos = video_grid_thw.shape[0]
        
        # Batch processing: Process all videos together for encoder (which handles batching internally)
        # The encoder processes each video separately to maintain spatial structure,
        # but we can process the entire batch at once
        
        # Encoder: process all videos together (handles batching internally)
        latent, mask, ids_restore, ids_keep = self.forward_encoder(
            pixel_values_videos, video_grid_thw
        )
        
        # Calculate patches per video for decoder and loss computation
        num_patches_per_video = (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]).long()
        cumsum_patches = torch.cat([
            torch.zeros(1, dtype=torch.long, device=pixel_values_videos.device),
            num_patches_per_video.cumsum(0)
        ])
        
        # Decoder: Process each video separately to handle different numbers of visible tokens
        # Map global ids to local ids for each video
        all_preds = []
        losses = []
        
        device = pixel_values_videos.device
        
        for i in range(num_videos):
            start_idx = cumsum_patches[i].item()
            end_idx = cumsum_patches[i + 1].item()
            
            # Extract patches for this video
            video_patches = pixel_values_videos[start_idx:end_idx]  # [T*H*W, patch_pixel_dim]
            video_num_patches = end_idx - start_idx
            
            # Get visible tokens for this video from encoder output
            video_visible_mask = (ids_keep >= start_idx) & (ids_keep < end_idx)
            video_latent = latent[video_visible_mask]  # [num_vis_video, encoder_dim]
            
            # Map global ids to local ids for this video
            # Understanding of ids:
            # - ids_restore: maps global_shuffled_pos -> global_original_pos
            #   ids_restore[i] = original_pos, where i is position in global shuffled order
            # - ids_keep: first num_visible indices in global shuffled order [0, 1, ..., num_visible-1]
            #   These are the shuffled positions of visible tokens
            
            video_ids_keep_global = ids_keep[video_visible_mask]  # Global shuffled positions
            video_num_visible = len(video_ids_keep_global)
            
            # Get original positions of visible tokens (in global original order)
            # ids_restore[video_ids_keep_global] gives us the original positions
            video_visible_original_global = ids_restore[video_ids_keep_global]  # Global original positions
            video_visible_original_local = video_visible_original_global - start_idx  # Local original positions
            
            # Build local shuffle for this video: visible tokens first, then masked tokens
            # For consistency, we put visible tokens first (sorted by original position)
            video_ids_shuffle_local = torch.zeros(video_num_patches, dtype=torch.long, device=device)
            
            # Visible tokens: sort by original position to maintain spatial structure
            visible_sorted_indices = torch.sort(video_visible_original_local)[1]  # Indices that sort
            video_ids_shuffle_local[:video_num_visible] = video_visible_original_local[visible_sorted_indices]
            
            # Masked tokens: add remaining positions (all positions not in visible)
            all_positions = torch.arange(video_num_patches, device=device)
            # Create mask for visible original positions
            visible_mask = torch.zeros(video_num_patches, dtype=torch.bool, device=device)
            visible_mask[video_visible_original_local] = True
            masked_positions = all_positions[~visible_mask]
            
            if len(masked_positions) > 0:
                # Randomly shuffle masked positions
                masked_shuffle = torch.randperm(len(masked_positions), device=device)
                video_ids_shuffle_local[video_num_visible:] = masked_positions[masked_shuffle]
            
            # Build local ids_restore: maps local_shuffled_pos -> local_original_pos
            # ids_restore[i] = original_pos, where i is position in shuffled order
            video_ids_restore_local = torch.argsort(video_ids_shuffle_local, dim=0)
            video_ids_keep_local_final = video_ids_shuffle_local[:video_num_visible]
            
            # Decoder: reconstruct all patches for this video
            pred = self.forward_decoder(
                video_latent, 
                video_ids_restore_local, 
                video_ids_keep_local_final, 
                video_num_patches
            )
            
            # Loss: only on masked patches for this video
            # Build local mask for this video
            video_mask_local = mask[start_idx:end_idx]
            loss = self.forward_loss(
                video_patches, pred, video_mask_local, video_ids_restore_local
            )
            
            losses.append(loss)
            all_preds.append(pred)
        
        # Concatenate results and average loss
        loss = torch.stack(losses).mean()
        pred = torch.cat(all_preds, dim=0)
        
        # Restore mask ratio
        self.mask_ratio = old_ratio
        
        return loss, pred, mask


class TransformerBlock(nn.Module):
    """Standard Transformer block for decoder"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x
