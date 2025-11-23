"""
MAE (Masked Autoencoder) for InternVL vision encoder
Based on Qwen's MAE implementation, adapted for InternVL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from internvl.model.internvl_chat import InternVisionModel, InternVisionConfig

# Try to import dist for distributed training
try:
    import torch.distributed as dist
except ImportError:
    dist = None


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


class InternViTMAE(nn.Module):
    """
    MAE for InternVL vision encoder
    
    Key design:
    - Encoder: 只处理visible tokens (未被mask的patches)，节省计算
    - Decoder: 轻量级，负责重建masked patches
    - Loss: 只在masked patches上计算
    
    Input format (compatible with InternVL):
    - pixel_values_videos: [N, patch_pixel_dim] - flattened video patches
    - video_grid_thw: [num_videos, 3] - (T, H, W) for each video
    """
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for decoder blocks to save memory"""
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
        model_id: str = "OpenGVLab/InternVL2_5-2B",
        vision_path: Optional[str] = None,
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
        
        # Load InternVL vision model
        print(f"Loading InternVL vision model from {vision_path or model_id}...")
        if vision_path:
            vision_config = InternVisionConfig.from_pretrained(vision_path)
            self.visual = InternVisionModel.from_pretrained(
                vision_path,
                torch_dtype=torch.float32,
                config=vision_config
            )
        else:
            # Try to load from model_id (may need to extract vision model from full model)
            vision_config = InternVisionConfig.from_pretrained(model_id)
            self.visual = InternVisionModel.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                config=vision_config
            )
        
        # Get device from vision model
        vision_device = next(self.visual.parameters()).device
        
        # Get key dimensions from InternVL's vision encoder
        self.encoder_embed_dim = vision_config.hidden_size  # e.g., 1408 for InternVL2.5-2B
        self.patch_size = vision_config.patch_size  # Typically 14 for InternVL
        # Note: patch_pixel_dim will be determined dynamically from input
        self.patch_pixel_dim = None  # Will be set dynamically
        
        # Masking parameters
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.spacetime_mask = spacetime_mask
        self.mask_strategy = mask_strategy
        self.mask_unit_size = mask_unit_size
        
        # ================== MAE Decoder (确保在正确设备上) ==================
        # Decoder input projection - 移到正确设备
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_embed_dim, bias=True).to(vision_device)
        
        # Mask token - 移到正确设备
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)).to(vision_device)
        
        # Decoder transformer blocks - 移到正确设备
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim, 
                decoder_num_heads, 
                mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            ).to(vision_device)  # 确保每个block在正确设备上
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim).to(vision_device)
        
        # Prediction head: predict RGB pixel values per patch
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masking strategy router - selects appropriate masking method based on self.mask_strategy
        
        Args:
            video_grid_thw: [num_videos, 3] - (T, H, W) for each video
            device: device for tensors
        
        Returns:
            mask: [total_patches] - binary mask (1 = masked, 0 = visible)
            ids_restore: [total_patches] - indices to restore original order
            ids_keep: [total_visible_patches] - indices of visible patches
            num_patches_per_video: [num_videos] - patches per video
            cumsum_patches: [num_videos+1] - cumulative patch indices
        """
        # Print masking strategy confirmation (only once)
        if not hasattr(self, '_mask_strategy_logged'):
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original random patch-level masking (each patch independently)"""
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
        
        return mask, ids_restore, ids_keep, num_patches_per_video, cumsum_patches
    
    def _tube_masking(
        self, 
        video_grid_thw: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            mask = spatial_mask.unsqueeze(0).repeat(T, 1).flatten()  # [T*H*W]
            
            # Get visible patch indices
            ids_keep = torch.where(~mask)[0]
            
            # Create restore indices
            ids_restore = torch.arange(num_patches, device=device)
            
            masks.append(mask)
            ids_restores.append(ids_restore)
            ids_keeps.append(ids_keep)
            num_patches_per_video_list.append(num_patches)
            cumsum_patches_list.append(cumsum_patches_list[-1] + num_patches)
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spatial block masking: mask spatial blocks (e.g., 4x4 patches) across all frames"""
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
            patch_mask_2d = patch_mask_2d[:H, :W]
            
            # Expand to all frames: [H, W] -> [T, H, W] -> [T*H*W]
            patch_mask = patch_mask_2d.unsqueeze(0).repeat(T, 1, 1).flatten()
            
            # Get visible patch indices
            ids_keep = torch.where(~patch_mask)[0]
            
            # Create restore indices
            ids_restore = torch.arange(T * H * W, device=device)
            
            masks.append(patch_mask)
            ids_restores.append(ids_restore)
            ids_keeps.append(ids_keep)
            num_patches_per_video_list.append(T * H * W)
            cumsum_patches_list.append(cumsum_patches_list[-1] + T * H * W)
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mask Unit (MU) masking similar to Hiera: mask 3D blocks (T x H x W)
        - Keeps both spatial and temporal locality
        - Best for sign language (hand gestures need both spatial structure and temporal motion)
        """
        num_videos = video_grid_thw.shape[0]
        block_h, block_w = self.mask_unit_size
        block_t = 2  # For temporal, use 2 frames per MU
        
        masks = []
        ids_restores = []
        ids_keeps = []
        num_patches_per_video_list = []
        cumsum_patches_list = [torch.zeros(1, dtype=torch.long, device=device)]
        
        for i in range(num_videos):
            T, H, W = video_grid_thw[i].tolist()
            
            # Calculate number of MUs in each dimension
            num_blocks_t = (T + block_t - 1) // block_t
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
        
        # Concatenate all videos
        mask = torch.cat(masks)
        ids_restore = torch.cat(ids_restores)
        ids_keep = torch.cat(ids_keeps)
        num_patches_per_video = torch.tensor(num_patches_per_video_list, device=device)
        cumsum_patches = torch.cat(cumsum_patches_list)
        
        return mask, ids_restore, ids_keep, num_patches_per_video, cumsum_patches
    
    def forward_encoder(
        self, 
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder forward pass - 只处理visible tokens!
        
        Args:
            pixel_values_videos: [N, patch_pixel_dim] - flattened patches (already extracted)
            video_grid_thw: [num_videos, 3] - (T, H, W) for each video (in patch units)
        
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
        
        # Step 2: Extract visible patches only
        visible_patches = pixel_values_videos[ids_keep]  # [num_visible, patch_pixel_dim]
        
        # Step 3: Convert patches to embeddings using linear projection
        # InternVL's patch_embed is designed for images, not patches
        # For MAE, we use a linear layer to project patches to embeddings
        # Get encoder device and dtype
        if hasattr(self.visual, 'encoder'):
            encoder_param = next(self.visual.encoder.parameters())
            encoder_device = encoder_param.device
            encoder_dtype = encoder_param.dtype
        else:
            encoder_device = device
            encoder_dtype = visible_patches.dtype
        
        if not hasattr(self, '_patch_embed_proj'):
            # 确保projection layer在正确的设备上
            self._patch_embed_proj = nn.Linear(
                self.patch_pixel_dim, 
                self.encoder_embed_dim
            ).to(device=encoder_device, dtype=encoder_dtype)  # 使用encoder的设备
            # Initialize weights
            nn.init.xavier_uniform_(self._patch_embed_proj.weight)
            if self._patch_embed_proj.bias is not None:
                nn.init.constant_(self._patch_embed_proj.bias, 0)
        
        # 确保输入也在正确设备上
        visible_patches_typed = visible_patches.to(device=encoder_device, dtype=encoder_dtype)
        visible_embeds = self._patch_embed_proj(visible_patches_typed)  # [num_visible, encoder_dim]
        
        # Step 4: Process through encoder blocks
        # InternVL's encoder expects [batch, seq_len, hidden_dim] format
        # Add batch dimension for encoder processing
        if hasattr(self.visual, 'encoder') and hasattr(self.visual.encoder, 'layers'):
            # Add batch dimension: [num_visible, encoder_dim] -> [1, num_visible, encoder_dim]
            x = visible_embeds.unsqueeze(0)  # [1, num_visible, encoder_dim]
            
            # InternVL's attention layers require bfloat16 input, but encoder weights are float32
            # Temporarily convert encoder to bfloat16 for computation
            # Store original dtype
            encoder_was_bfloat16 = encoder_param.dtype == torch.bfloat16
            
            # Convert encoder to bfloat16 if not already, ensuring device is preserved
            # Use .cuda() to ensure all submodules are on CUDA
            if not encoder_was_bfloat16:
                if not encoder_device.type == 'cuda':
                    self.visual.encoder = self.visual.encoder.cuda()
                self.visual.encoder = self.visual.encoder.to(dtype=torch.bfloat16)
            
            # Convert input to bfloat16, ensuring it's on CUDA
            if not x.is_cuda:
                x = x.cuda()
            x = x.to(dtype=torch.bfloat16)
            
            # Process through encoder layers
            for layer in self.visual.encoder.layers:
                x = layer(x)  # [1, num_visible, encoder_dim]
            
            # Restore original dtype if needed, ensuring device is preserved
            if not encoder_was_bfloat16:
                self.visual.encoder = self.visual.encoder.to(dtype=encoder_dtype)
                # Ensure encoder stays on CUDA
                if not next(self.visual.encoder.parameters()).is_cuda:
                    self.visual.encoder = self.visual.encoder.cuda()
            
            # Remove batch dimension: [1, num_visible, encoder_dim] -> [num_visible, encoder_dim]
            # Convert back to encoder dtype for consistency
            latent = x.squeeze(0).to(dtype=encoder_dtype)
        else:
            # Fallback: use embeddings directly
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
        """
        # Ensure latent is on the same device as decoder_embed
        decoder_device = next(self.decoder_embed.parameters()).device
        latent = latent.to(device=decoder_device)
        ids_restore = ids_restore.to(device=decoder_device)
        ids_keep = ids_keep.to(device=decoder_device)
        
        # Step 1: Project encoder output to decoder dimension
        x = self.decoder_embed(latent)  # [num_visible, decoder_dim]
        
        # Step 2: Build full sequence with visible tokens + mask tokens
        visible_positions_original = ids_restore[ids_keep]  # [num_visible]
        
        # Create full sequence [num_patches, decoder_dim]
        decoder_dim = x.shape[1]
        full_sequence = torch.zeros(num_patches, decoder_dim, device=x.device, dtype=x.dtype)
        
        # Place visible tokens at their original positions
        full_sequence[visible_positions_original] = x
        
        # Place mask tokens at remaining positions
        all_positions = torch.arange(num_patches, device=x.device)
        mask_positions_original = torch.ones(num_patches, dtype=torch.bool, device=x.device)
        mask_positions_original[visible_positions_original] = False
        mask_positions_original = all_positions[mask_positions_original]
        num_mask = len(mask_positions_original)
        if num_mask > 0:
            full_sequence[mask_positions_original] = self.mask_token.squeeze(0).expand(num_mask, -1)
        
        x = full_sequence
        
        # Step 3: Apply decoder blocks
        if getattr(self, '_gradient_checkpointing', False) and self.training:
            from torch.utils.checkpoint import checkpoint
            checkpoint_kwargs = getattr(self, '_gradient_checkpointing_kwargs', {})
            use_reentrant = checkpoint_kwargs.get('use_reentrant', True)
            for blk in self.decoder_blocks:
                x = checkpoint(blk, x, use_reentrant=use_reentrant)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)
        
        # Step 4: Predictor projection
        if self.decoder_pred is None:
            raise RuntimeError("decoder_pred not initialized")
        pred = self.decoder_pred(x)  # [num_patches, patch_pixel_dim]
        
        return pred
    
    def forward_loss(
        self, 
        pixel_values_videos: torch.Tensor,
        pred: torch.Tensor, 
        mask: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss only on masked patches"""
        target = pixel_values_videos
        
        # Normalize target patches if enabled
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # Compute MSE loss per patch
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        # Only compute loss on masked patches
        loss = (loss * mask.float()).sum() / mask.sum().float()
        
        return loss
    
    def _ensure_patch_pixel_dim(self, pixel_values_videos: torch.Tensor):
        """Ensure patch_pixel_dim is set from actual input dimension"""
        actual_dim = pixel_values_videos.shape[-1]
        device = pixel_values_videos.device
        model_dtype = next(self.decoder_embed.parameters()).dtype
        
        if self.patch_pixel_dim is None:
            self.patch_pixel_dim = actual_dim
            decoder_embed_dim = self.decoder_embed.out_features
            
            if self.decoder_pred is None:
                self.decoder_pred = nn.Linear(
                    decoder_embed_dim,
                    self.patch_pixel_dim,
                    bias=True
                ).to(device=device, dtype=model_dtype)
                torch.nn.init.xavier_uniform_(self.decoder_pred.weight)
                if self.decoder_pred.bias is not None:
                    nn.init.constant_(self.decoder_pred.bias, 0)
                self.add_module('decoder_pred', self.decoder_pred)
                
                if not hasattr(torch.distributed, 'is_initialized') or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"Detected patch_pixel_dim: {self.patch_pixel_dim} (creating decoder_pred layer with dtype {model_dtype})")
        elif self.patch_pixel_dim != actual_dim:
            raise ValueError(f"Patch pixel dimension mismatch: expected {self.patch_pixel_dim}, got {actual_dim}")
        
        if self.decoder_pred is not None:
            if self.decoder_pred.out_features != actual_dim:
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
        """
        self._ensure_patch_pixel_dim(pixel_values_videos)
        
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        old_ratio = self.mask_ratio
        self.mask_ratio = mask_ratio
        
        num_videos = video_grid_thw.shape[0]
        
        # Encoder
        latent, mask, ids_restore, ids_keep = self.forward_encoder(
            pixel_values_videos, video_grid_thw
        )
        
        # Calculate patches per video for decoder and loss computation
        num_patches_per_video = (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]).long()
        cumsum_patches = torch.cat([
            torch.zeros(1, dtype=torch.long, device=pixel_values_videos.device),
            num_patches_per_video.cumsum(0)
        ])
        
        # Decoder: Process each video separately
        all_preds = []
        losses = []
        device = pixel_values_videos.device
        
        for i in range(num_videos):
            start_idx = cumsum_patches[i].item()
            end_idx = cumsum_patches[i + 1].item()
            
            video_patches = pixel_values_videos[start_idx:end_idx]
            video_num_patches = end_idx - start_idx
            
            # Get visible tokens for this video
            # ids_keep contains indices in the shuffled sequence (global)
            # We need to find which of these belong to the current video
            # First, convert ids_keep (shuffled indices) to original positions using ids_restore
            ids_keep_original = ids_restore[ids_keep]  # [num_visible,]
            
            # Now find which visible patches belong to this video (in original order)
            video_visible_mask = (ids_keep_original >= start_idx) & (ids_keep_original < end_idx)
            video_latent = latent[video_visible_mask]
            
            video_ids_keep_original_global = ids_keep_original[video_visible_mask]
            video_num_visible = len(video_ids_keep_original_global)
            
            if video_num_visible == 0:
                # No visible patches for this video
                pred = torch.zeros(video_num_patches, self.patch_pixel_dim, device=device, dtype=pixel_values_videos.dtype)
                video_mask_local = mask[start_idx:end_idx]
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                losses.append(loss)
                all_preds.append(pred)
                continue
            
            # Convert global original indices to local indices
            video_visible_original_local = video_ids_keep_original_global - start_idx
            
            # Get the mask for this video (in original order)
            video_mask_local = mask[start_idx:end_idx]
            
            # Build shuffle and restore indices for this video
            # We need to create a shuffle that places visible patches first, then masked patches
            video_ids_shuffle_local = torch.zeros(video_num_patches, dtype=torch.long, device=device)
            
            # Place visible patches at the beginning (sorted by their original positions)
            visible_sorted_indices = torch.sort(video_visible_original_local)[1]
            video_ids_shuffle_local[:video_num_visible] = video_visible_original_local[visible_sorted_indices]
            
            # Place masked patches after visible ones
            all_positions = torch.arange(video_num_patches, device=device)
            visible_mask_local = torch.zeros(video_num_patches, dtype=torch.bool, device=device)
            visible_mask_local[video_visible_original_local] = True
            masked_positions = all_positions[~visible_mask_local]
            
            if len(masked_positions) > 0:
                masked_shuffle = torch.randperm(len(masked_positions), device=device)
                video_ids_shuffle_local[video_num_visible:] = masked_positions[masked_shuffle]
            
            # Create restore mapping
            video_ids_restore_local = torch.argsort(video_ids_shuffle_local, dim=0)
            video_ids_keep_local_final = video_ids_shuffle_local[:video_num_visible]
            
            # Decoder
            pred = self.forward_decoder(
                video_latent, 
                video_ids_restore_local,
                video_ids_keep_local_final,
                video_num_patches
            )
            
            # Loss
            loss = self.forward_loss(
                video_patches, pred, video_mask_local, video_ids_restore_local
            )
            
            losses.append(loss)
            all_preds.append(pred)
        
        # Concatenate results and average loss
        loss = torch.stack(losses).mean()
        pred = torch.cat(all_preds, dim=0)
        
        self.mask_ratio = old_ratio
        
        return loss, pred, mask

