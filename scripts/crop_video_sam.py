#!/usr/bin/env python3
"""
SAM-Based Video Segmentation for Sign Language
Uses Segment Anything Model (SAM) to automatically segment upper body
and mask background for sign language videos.
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import sys
import torch
import multiprocessing
from functools import partial

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything not available.")
    print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Will use center-point prompt only.")

# Try to import Mask2Former
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.model_zoo import model_zoo
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    MASK2FORMER_AVAILABLE = True
except ImportError:
    MASK2FORMER_AVAILABLE = False
    print("Warning: Detectron2/Mask2Former not available. Install with: pip install detectron2")


class SegmentationBackend:
    """Base class for different segmentation backends."""
    
    def segment(self, rgb_frame, bbox=None):
        """
        Segment a frame.
        
        Args:
            rgb_frame: RGB frame
            bbox: Optional bounding box (x, y, width, height)
            
        Returns:
            np.ndarray: Binary mask (255 for foreground, 0 for background)
        """
        raise NotImplementedError


class MediaPipeSegmentation(SegmentationBackend):
    """MediaPipe Selfie Segmentation backend - fast and lightweight."""
    
    def __init__(self, device="cuda", model_selection=1, mask_threshold=0.5):
        """
        Initialize MediaPipe Selfie Segmentation.
        
        Args:
            device: Device (not used for MediaPipe, kept for compatibility)
            model_selection: 0=general model, 1=landscape model (better quality)
            mask_threshold: Threshold for converting probability mask to binary (0.0-1.0)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for this backend")
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.model_selection = model_selection
        self.mask_threshold = mask_threshold
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection  # 0=general, 1=landscape (better quality)
        )
    
    def segment(self, rgb_frame, bbox=None):
        """Segment person using MediaPipe Selfie Segmentation."""
        results = self.selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask
        # Convert to binary mask using configurable threshold
        mask = (mask > self.mask_threshold).astype(np.uint8) * 255
        
        # Post-process mask for smoother edges and better quality (same as SAM)
        # Fill small holes
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Remove small noise/artifacts
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Smooth edges with larger kernel
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        
        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (7, 7), 1.0)
        
        # Re-threshold to maintain binary mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask


class SAMVideoProcessor:
    """
    SAM-Based Video Processor with Complete Pipeline:
    
    Pipeline: Input Frame → Detection → SAM Segmentation → Masking → Output Frame
    
    Steps:
    1. detect_upper_body_region(): Detects upper body region (MediaPipe or center-based)
    2. segment_with_sam(): SAM segments the region automatically
    3. apply_mask(): Applies mask to frame (black background)
    4. Optional: visualize_detections(): Draws segmentation overlay
    
    Uses Segment Anything Model (SAM) for precise segmentation.
    """
    
    def __init__(self, sam_checkpoint=None, model_type="vit_h", device="cuda", 
                 use_automatic_generator=True, stability_threshold=0.95, 
                 temporal_smoothing=0.6, remove_clothing=True, 
                 segmentation_backend="sam",
                 crop_scale=1.0, mediapipe_model_selection=1, 
                 mediapipe_mask_threshold=0.5, mediapipe_pose_confidence=0.5,
                 mediapipe_hands_confidence=0.5, mediapipe_face_confidence=0.5,
                 mediapipe_face_expand=15, mediapipe_hand_margin=25,
                 mediapipe_face_shrink=0.2,
                 output_size=None,
                 background_color="black", 
                 use_blurred_background=False, background_blur_size=51, background_brightness=0.1,
                 face_hands_only=False):
        """
        Initialize video processor with multiple segmentation backend options.
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint (required for SAM backends)
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            device: Device to use ("cuda" or "cpu")
            use_automatic_generator: Use automatic mask generator (better quality, SAM only)
            background_color: Background color for masked regions ("black", "gray", or "blurred"). 
                             "blurred" uses blurred original background (best for ViT attention).
            use_blurred_background: If True, use blurred original background instead of solid color.
            background_blur_size: Gaussian blur kernel size for blurred background (default: 51).
            background_brightness: Brightness multiplier for blurred background (0.0-1.0, default: 0.1).
            stability_threshold: Stability score threshold for mask filtering
            temporal_smoothing: Temporal smoothing factor (0.0-1.0). Higher = more smoothing.
                               0.5 means 50% previous frame + 50% current frame (default: 0.5)
            remove_clothing: If True, remove clothing and keep only face, hands, exposed skin
            face_hands_only: If True, only extract face and hands (no body parts). Requires MediaPipe.
            segmentation_backend: Which backend to use:
                                  - "sam": Standard SAM (default)
                                  - "mediapipe": MediaPipe Selfie Segmentation (fastest)
            crop_scale: Scale factor for bounding box size (1.0 = default, < 1.0 = smaller crop, > 1.0 = larger crop)
            mediapipe_model_selection: MediaPipe model (0=general, 1=landscape) for selfie segmentation
            mediapipe_mask_threshold: Threshold for MediaPipe mask (0.0-1.0, lower = more permissive)
            mediapipe_pose_confidence: Confidence threshold for MediaPipe pose detection (0.0-1.0)
            mediapipe_hands_confidence: Confidence threshold for MediaPipe hands detection (0.0-1.0)
            mediapipe_face_confidence: Confidence threshold for MediaPipe face detection (0.0-1.0)
            mediapipe_face_expand: Extra pixels to expand around detected face box (default: 15)
            mediapipe_hand_margin: Extra pixels to expand around detected hand landmarks (default: 25)
            mediapipe_face_shrink: Fraction (0-1) to trim from bottom of face box to avoid neck (default: 0.2)
            output_size: If set (int), resize final masked frames to output_size x output_size (e.g., 320 or 224)
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.background_color = background_color.lower()
        self.use_blurred_background = use_blurred_background or (background_color.lower() == "blurred")
        self.background_blur_size = background_blur_size
        self.background_brightness = background_brightness
        self.use_automatic_generator = use_automatic_generator
        self.stability_threshold = stability_threshold
        self.temporal_smoothing = temporal_smoothing
        self.remove_clothing = remove_clothing
        self.face_hands_only = face_hands_only
        self.segmentation_backend_name = segmentation_backend
        self.crop_scale = crop_scale
        self.mediapipe_face_expand = mediapipe_face_expand
        self.mediapipe_hand_margin = mediapipe_hand_margin
        self.mediapipe_face_shrink = mediapipe_face_shrink
        self.output_size = output_size
        
        # Initialize temporal smoothing buffers
        self.previous_mask = None
        self.previous_bbox = None
        self.smoothing_alpha = temporal_smoothing  # Weight for previous frame
        
        # Initialize segmentation backend
        print(f"🚀 Loading segmentation backend: {segmentation_backend}...")
        
        if segmentation_backend == "mediapipe":
            if not MEDIAPIPE_AVAILABLE:
                raise ImportError("MediaPipe is required for mediapipe backend")
            self.segmentation_backend = MediaPipeSegmentation(
                device=self.device,
                model_selection=mediapipe_model_selection,
                mask_threshold=mediapipe_mask_threshold
            )
            self.sam = None
            self.predictor = None
            self.mask_generator = None
            print(f"✅ MediaPipe Selfie Segmentation loaded! (model={mediapipe_model_selection}, threshold={mediapipe_mask_threshold})")
            
        elif segmentation_backend == "sam" or segmentation_backend is None:
            # Default: Standard SAM
            if not SAM_AVAILABLE:
                raise ImportError(
                    "Segment Anything Model is required.\n"
                    "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                    "Download checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )
            if sam_checkpoint is None:
                raise ValueError(
                    "sam_checkpoint is required for SAM backend.\n"
                    "Use --segmentation-backend mediapipe to skip SAM checkpoint requirement."
                )
            
            print(f"🚀 Loading SAM model ({model_type}) on {self.device}...")
            
            try:
                self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                self.sam.to(device=self.device)
                self.sam.eval()
                
                if use_automatic_generator:
                    # Use automatic mask generator for better quality
                    # Reduced points_per_side from 32 to 16 for faster processing
                    # (32->16 reduces computation by ~4x: 32*32=1024 vs 16*16=256 points)
                    print("⚠️  Using automatic mask generator (slower but better quality)")
                    print("   If too slow, use --no-automatic for faster prompt-based mode")
                    self.mask_generator = SamAutomaticMaskGenerator(
                        self.sam,
                        points_per_side=16,  # Reduced from 32 for speed
                        pred_iou_thresh=0.88,  # Slightly lower for faster filtering
                        stability_score_thresh=stability_threshold,
                        crop_n_layers=1,
                        crop_n_points_downscale_factor=2,
                        min_mask_region_area=100,
                    )
                    self.predictor = None  # Not used in automatic mode
                else:
                    # Use predictor for prompt-based segmentation (much faster)
                    print("⚡ Using prompt-based segmentation (faster)")
                    self.predictor = SamPredictor(self.sam)
                    self.mask_generator = None  # Not used in prompt mode
                
                self.segmentation_backend = None  # Use legacy SAM code
                print("✅ SAM model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading SAM model: {e}")
                raise
        else:
            raise ValueError(f"Unknown segmentation backend: {segmentation_backend}")
        
        # Initialize MediaPipe for upper body detection (optional)
        self.mediapipe_pose_confidence = mediapipe_pose_confidence
        self.mediapipe_hands_confidence = mediapipe_hands_confidence
        self.mediapipe_face_confidence = mediapipe_face_confidence
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=self.mediapipe_pose_confidence,
                min_tracking_confidence=self.mediapipe_pose_confidence,
                model_complexity=1
            )
            # Add hands and face detection for clothing removal
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.mediapipe_hands_confidence,
                min_tracking_confidence=self.mediapipe_hands_confidence
            )
            self.mp_face = mp.solutions.face_detection
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.mediapipe_face_confidence
            )
            self.use_mediapipe = True
        else:
            self.use_mediapipe = False
    
    def detect_upper_body_region(self, rgb_frame):
        """
        STEP 1: Detect Upper Body Region
        Detects upper body bounding box for SAM prompt with temporal smoothing.
        
        Args:
            rgb_frame: RGB frame from video
            
        Returns:
            tuple: (x, y, width, height) bounding box or None
        """
        h, w = rgb_frame.shape[:2]
        current_bbox = None
        
        if self.use_mediapipe:
            # Use MediaPipe pose to detect upper body
            pose_results = self.pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                # Upper body keypoints (shoulders, head, arms)
                upper_body_indices = [0, 2, 3, 7, 8, 11, 12, 13, 14, 15, 16]
                xs = []
                ys = []
                
                for idx in upper_body_indices:
                    if idx < len(pose_results.pose_landmarks.landmark):
                        landmark = pose_results.pose_landmarks.landmark[idx]
                        if landmark.visibility > 0.5:
                            xs.append(landmark.x * w)
                            ys.append(landmark.y * h)
                
                if xs and ys:
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))
                    
                    # Add margin (scaled by crop_scale)
                    margin_x = int((x_max - x_min) * 0.1 * self.crop_scale)
                    margin_y = int((y_max - y_min) * 0.1 * self.crop_scale)
                    
                    x = max(0, x_min - margin_x)
                    y = max(0, y_min - margin_y)
                    width = min(w - x, x_max - x_min + 2 * margin_x)
                    height = min(h - y, y_max - y_min + 2 * margin_y)
                    
                    current_bbox = (x, y, width, height)
        
        if current_bbox is None:
            # Fallback: use center region (assumes signer is centered)
            # For sign language videos, person is typically in upper-center
            center_x = w // 2
            center_y = int(h * 0.35)  # Upper third (for upper body)
            # Apply crop_scale to default sizes
            bbox_width = int(w * 0.7 * self.crop_scale)  # 70% of width (wider for arms) * scale
            bbox_height = int(h * 0.65 * self.crop_scale)  # 65% of height (upper body) * scale
            
            x = center_x - bbox_width // 2
            y = center_y - bbox_height // 3  # More space above than below
            current_bbox = (x, y, bbox_width, bbox_height)
        
        # Apply temporal smoothing to bounding box
        if self.previous_bbox is not None and self.temporal_smoothing > 0:
            prev_x, prev_y, prev_w, prev_h = self.previous_bbox
            curr_x, curr_y, curr_w, curr_h = current_bbox
            
            # Smooth coordinates with exponential moving average
            smooth_x = int(self.smoothing_alpha * prev_x + (1 - self.smoothing_alpha) * curr_x)
            smooth_y = int(self.smoothing_alpha * prev_y + (1 - self.smoothing_alpha) * curr_y)
            smooth_w = int(self.smoothing_alpha * prev_w + (1 - self.smoothing_alpha) * curr_w)
            smooth_h = int(self.smoothing_alpha * prev_h + (1 - self.smoothing_alpha) * curr_h)
            
            # Ensure smoothed bbox stays within frame bounds
            smooth_x = max(0, min(smooth_x, w - 1))
            smooth_y = max(0, min(smooth_y, h - 1))
            smooth_w = max(10, min(smooth_w, w - smooth_x))
            smooth_h = max(10, min(smooth_h, h - smooth_y))
            
            smoothed_bbox = (smooth_x, smooth_y, smooth_w, smooth_h)
            self.previous_bbox = smoothed_bbox
            return smoothed_bbox
        else:
            self.previous_bbox = current_bbox
            return current_bbox
    
    def segment_with_sam(self, rgb_frame, bbox=None, point_prompt=None):
        """
        STEP 2: Segmentation Pipeline
        Segments the upper body region using selected backend.
        
        Args:
            rgb_frame: RGB frame from video
            bbox: Optional bounding box (x, y, width, height) as prompt
            point_prompt: Optional center point as prompt
            
        Returns:
            np.ndarray: Binary mask (255 for foreground, 0 for background)
        """
        h, w = rgb_frame.shape[:2]
        
        # Use backend if available (MediaPipe)
        if self.segmentation_backend is not None:
            mask = self.segmentation_backend.segment(rgb_frame, bbox=bbox)
            
            # Apply temporal smoothing for MediaPipe backend (same as SAM)
            if self.previous_mask is not None and self.temporal_smoothing > 0:
                # Check if previous mask has same dimensions
                if self.previous_mask.shape == mask.shape:
                    # Exponential moving average for mask smoothing
                    mask_float = mask.astype(np.float32) / 255.0
                    prev_mask_float = self.previous_mask.astype(np.float32) / 255.0
                    smoothed_mask = (self.smoothing_alpha * prev_mask_float + 
                                   (1 - self.smoothing_alpha) * mask_float)
                    mask = (smoothed_mask * 255).astype(np.uint8)
                
                self.previous_mask = mask.copy()
            else:
                self.previous_mask = mask.copy()
            
            return mask
        
        # Legacy SAM code (for standard SAM backend)
        if self.use_automatic_generator:
            # Automatic mask generation (better quality, slower)
            # Note: This can be very slow (several seconds per frame)
            masks = self.mask_generator.generate(rgb_frame)
            
            if not masks:
                # Fallback: return empty mask
                return np.zeros((h, w), dtype=np.uint8)
            
            # Select the best mask using multiple criteria
            # Filter by stability score first
            valid_masks = [m for m in masks if m.get('stability_score', 0) >= self.stability_threshold]
            
            if not valid_masks:
                # Lower threshold if no masks meet it
                valid_masks = [m for m in masks if m.get('stability_score', 0) >= max(0.7, self.stability_threshold - 0.1)]
            
            if not valid_masks:
                # Use all masks if still none meet threshold
                valid_masks = masks
            
            if not valid_masks:
                # Still no masks, return empty
                return np.zeros((h, w), dtype=np.uint8)
            
            # Score masks using multiple criteria:
            # 1. Area (larger is better, but not too large - person should be reasonable size)
            # 2. Stability score
            # 3. Overlap with bounding box if available
            # 4. Position (center-biased for sign language videos)
            frame_area = h * w
            best_mask = None
            best_score = -1
            
            for mask_data in valid_masks:
                mask_seg = mask_data['segmentation']
                area = mask_data['area']
                stability = mask_data.get('stability_score', 0)
                pred_iou = mask_data.get('predicted_iou', 0)
                
                # Calculate score components
                # 1. Area score: prefer masks that are 5-50% of frame (reasonable person size)
                area_ratio = area / frame_area
                if area_ratio < 0.01:  # Too small
                    area_score = 0
                elif area_ratio > 0.7:  # Too large (likely background)
                    area_score = 0
                elif 0.05 <= area_ratio <= 0.5:  # Ideal range
                    area_score = 1.0
                else:
                    area_score = 1.0 - abs(area_ratio - 0.2) * 2  # Falloff outside ideal
                
                # 2. Stability and IoU scores
                quality_score = (stability * 0.6 + pred_iou * 0.4)
                
                # 3. Bbox overlap score if bbox provided
                bbox_score = 1.0
                if bbox:
                    x, y, width, height = bbox
                    bbox_mask = np.zeros((h, w), dtype=bool)
                    bbox_mask[y:y+height, x:x+width] = True
                    overlap = np.sum(mask_seg & bbox_mask) / max(1, np.sum(mask_seg))
                    bbox_score = overlap
                
                # 4. Center position score (people usually centered in sign language videos)
                mask_y_indices, mask_x_indices = np.where(mask_seg)
                if len(mask_y_indices) > 0:
                    center_y = np.mean(mask_y_indices) / h
                    center_x = np.mean(mask_x_indices) / w
                    # Prefer masks centered horizontally (0.3-0.7 of width) and upper-middle vertically
                    center_score = (1.0 - abs(center_x - 0.5) * 2) * (1.0 - abs(center_y - 0.4) * 1.5)
                    center_score = max(0, center_score)
                else:
                    center_score = 0
                
                # Combined score (weighted combination)
                total_score = (
                    area_score * 0.3 +
                    quality_score * 0.4 +
                    bbox_score * 0.2 +
                    center_score * 0.1
                )
                
                if total_score > best_score:
                    best_score = total_score
                    best_mask = mask_data
            
            # Fallback: if no mask found (shouldn't happen), use largest
            if best_mask is None:
                best_mask = max(valid_masks, key=lambda m: m['area'])
            
            # Convert to binary mask
            mask = best_mask['segmentation'].astype(np.uint8) * 255
            
            # Apply temporal smoothing to mask (for automatic mode)
            if self.previous_mask is not None and self.temporal_smoothing > 0:
                # Check if previous mask has same dimensions
                if self.previous_mask.shape == mask.shape:
                    # Exponential moving average for mask smoothing
                    mask_float = mask.astype(np.float32) / 255.0
                    prev_mask_float = self.previous_mask.astype(np.float32) / 255.0
                    smoothed_mask = (self.smoothing_alpha * prev_mask_float + 
                                   (1 - self.smoothing_alpha) * mask_float)
                    mask = (smoothed_mask * 255).astype(np.uint8)
                
                self.previous_mask = mask.copy()
            else:
                self.previous_mask = mask.copy()
            
        else:
            # Prompt-based segmentation (faster, requires prompts)
            self.predictor.set_image(rgb_frame)
            
            if bbox:
                # Use bounding box as prompt
                x, y, width, height = bbox
                box = np.array([x, y, x + width, y + height])
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=True,
                )
            elif point_prompt:
                # Use center point as prompt
                point_coords = np.array([point_prompt])
                point_labels = np.array([1])  # 1 = foreground point
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            else:
                # Default: center point
                center = (w // 2, h // 2)
                point_coords = np.array([center])
                point_labels = np.array([1])
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            
            # Select the best mask using multiple criteria
            best_mask_idx = 0
            best_score = -1
            
            for idx, (mask_candidate, score, logit) in enumerate(zip(masks, scores, logits)):
                mask_binary = mask_candidate.astype(np.uint8)
                
                # Calculate additional quality metrics
                mask_area = np.sum(mask_binary)
                frame_area = h * w
                area_ratio = mask_area / frame_area
                
                # Penalize masks that are too small (< 1%) or too large (> 70%)
                if area_ratio < 0.01 or area_ratio > 0.7:
                    continue  # Skip obviously bad masks
                
                # If bbox provided, calculate overlap
                bbox_score = 1.0
                if bbox:
                    x, y, width, height = bbox
                    bbox_mask = np.zeros((h, w), dtype=bool)
                    bbox_mask[y:y+height, x:x+width] = True
                    overlap = np.sum(mask_binary & bbox_mask) / max(1, mask_area)
                    bbox_score = overlap
                    # Prefer masks with good bbox overlap
                    if overlap < 0.3:
                        continue  # Skip masks with poor bbox overlap
                
                # Combined score: model score + area appropriateness + bbox overlap
                area_score = 1.0 if 0.05 <= area_ratio <= 0.5 else 0.8
                combined_score = score * 0.5 + area_score * 0.2 + bbox_score * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_mask_idx = idx
            
            # Fallback: if no mask passed filters, use highest scoring one
            if best_score == -1:
                best_mask_idx = np.argmax(scores)
            
            mask = masks[best_mask_idx].astype(np.uint8) * 255
            
            # Apply temporal smoothing to mask (for prompt-based mode)
            if self.previous_mask is not None and self.temporal_smoothing > 0:
                # Check if previous mask has same dimensions
                if self.previous_mask.shape == mask.shape:
                    # Exponential moving average for mask smoothing
                    mask_float = mask.astype(np.float32) / 255.0
                    prev_mask_float = self.previous_mask.astype(np.float32) / 255.0
                    smoothed_mask = (self.smoothing_alpha * prev_mask_float + 
                                   (1 - self.smoothing_alpha) * mask_float)
                    mask = (smoothed_mask * 255).astype(np.uint8)
                
                self.previous_mask = mask.copy()
            else:
                self.previous_mask = mask.copy()
        
        # Post-process mask for smoother edges and better quality
        # First, fill small holes in the mask
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Remove small noise/artifacts
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Smooth edges with larger kernel
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        
        # Apply Gaussian blur for smoother edges (reduces aliasing)
        mask = cv2.GaussianBlur(mask, (7, 7), 1.0)
        
        # Re-threshold to maintain binary mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Optional: dilate slightly to ensure we don't cut off edges
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        return mask
    
    def detect_body_parts_mask(self, rgb_frame, sam_mask):
        """
        Detect and create mask for body parts only (face, hands, arms skin) excluding clothing.
        
        Args:
            rgb_frame: RGB frame
            sam_mask: Binary mask from SAM (person segmentation)
            
        Returns:
            np.ndarray: Binary mask for body parts only (255 for skin/body parts, 0 for clothing/background)
        """
        h, w = rgb_frame.shape[:2]
        body_parts_mask = np.zeros((h, w), dtype=np.uint8)
        
        if not self.use_mediapipe:
            # If MediaPipe not available, use simple skin color detection
            return self.skin_color_detection(rgb_frame, sam_mask)
        
        # Convert RGB to BGR for OpenCV operations
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Detect and add face
        face_results = self.face_detector.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                # Expand slightly to include neck area
                expand = 10
                x = max(0, x - expand)
                y = max(0, y - expand)
                width = min(w - x, width + 2 * expand)
                height = min(h - y, height + 2 * expand)
                body_parts_mask[y:y+height, x:x+width] = 255
        
        # 2. Detect and add hands
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get bounding box of hand landmarks
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                # Expand hand region
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                body_parts_mask[y_min:y_max, x_min:x_max] = 255
        
        # 3. Detect arms/neck using pose landmarks + skin color detection
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            # Keypoints for arms and neck
            # Left/Right shoulders, elbows, wrists, and neck area
            keypoint_indices = {
                'left_arm': [11, 13, 15],  # Left shoulder, elbow, wrist
                'right_arm': [12, 14, 16],  # Right shoulder, elbow, wrist
                'neck': [0, 2, 5, 7],  # Nose, eyes, ears (for neck region)
            }
            
            landmarks = pose_results.pose_landmarks.landmark
            arm_points = []
            
            # Collect arm keypoints
            for idx in keypoint_indices['left_arm'] + keypoint_indices['right_arm']:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    if lm.visibility > 0.5:
                        arm_points.append((int(lm.x * w), int(lm.y * h)))
            
            # Create arm regions using skin color detection
            if arm_points:
                # Define arm region bounding box
                if arm_points:
                    xs = [p[0] for p in arm_points]
                    ys = [p[1] for p in arm_points]
                    x_min, x_max = max(0, min(xs) - 30), min(w, max(xs) + 30)
                    y_min, y_max = max(0, min(ys) - 30), min(h, max(ys) + 30)
                    
                    # Use skin color detection in arm region
                    arm_region = frame_bgr[y_min:y_max, x_min:x_max]
                    if arm_region.size > 0:
                        arm_skin_mask = self.skin_color_detection_small(arm_region)
                        # Only keep skin regions that are within SAM mask
                        arm_sam_mask = sam_mask[y_min:y_max, x_min:x_max]
                        arm_skin_mask = arm_skin_mask & (arm_sam_mask > 0)
                        body_parts_mask[y_min:y_max, x_min:x_max] = np.maximum(
                            body_parts_mask[y_min:y_max, x_min:x_max],
                            arm_skin_mask.astype(np.uint8) * 255
                        )
        
        # 4. Use skin color detection for exposed skin areas within SAM mask
        # This helps catch neck and other exposed skin areas
        skin_mask = self.skin_color_detection(rgb_frame, sam_mask)
        
        # Combine: body parts mask + skin mask (within SAM mask)
        combined_mask = np.maximum(body_parts_mask, skin_mask)
        combined_mask = combined_mask & (sam_mask > 0)  # Only keep within SAM person mask
        
        return combined_mask
    
    def detect_face_hands_only_mask(self, rgb_frame):
        """
        Detect and create mask for face and hands only (no body parts).
        This method only uses MediaPipe face and hand detection.
        
        Args:
            rgb_frame: RGB frame
            
        Returns:
            np.ndarray: Binary mask for face and hands only (255 for face/hands, 0 for background)
        """
        h, w = rgb_frame.shape[:2]
        face_hands_mask = np.zeros((h, w), dtype=np.uint8)
        
        if not self.use_mediapipe:
            raise ValueError("face_hands_only mode requires MediaPipe. Please install: pip install mediapipe")
        
        # 1. Detect and add face
        face_results = self.face_detector.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Apply face expansion and shrink settings
                expand_x = self.mediapipe_face_expand
                expand_y = self.mediapipe_face_expand
                
                # Shrink from bottom to avoid neck region
                if self.mediapipe_face_shrink > 0:
                    height = int(height * (1 - self.mediapipe_face_shrink))
                
                x = max(0, x - expand_x)
                y = max(0, y - expand_y)
                width = min(w - x, width + 2 * expand_x)
                height = min(h - y, height + 2 * expand_y)
                
                face_hands_mask[y:y+height, x:x+width] = 255
        
        # 2. Detect and add hands
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get bounding box of hand landmarks
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                # Apply hand margin setting
                margin = self.mediapipe_hand_margin
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                face_hands_mask[y_min:y_max, x_min:x_max] = 255
        
        # Post-process mask for smoother edges
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        face_hands_mask = cv2.morphologyEx(face_hands_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        face_hands_mask = cv2.morphologyEx(face_hands_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Smooth edges
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        face_hands_mask = cv2.morphologyEx(face_hands_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        face_hands_mask = cv2.GaussianBlur(face_hands_mask, (7, 7), 1.0)
        _, face_hands_mask = cv2.threshold(face_hands_mask, 127, 255, cv2.THRESH_BINARY)
        
        return face_hands_mask
    
    def skin_color_detection(self, rgb_frame, sam_mask):
        """
        Detect skin color regions using HSV color space.
        
        Args:
            rgb_frame: RGB frame
            sam_mask: Binary mask to constrain search area
            
        Returns:
            np.ndarray: Binary mask of skin regions
        """
        # Convert to HSV for better skin detection
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # HSV ranges for skin detection (works for various skin tones)
        # Lower bound: hue 0-20, saturation > 40, value > 50
        # Upper bound: hue 0-20, saturation < 255, value < 255
        lower_skin1 = np.array([0, 40, 50], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        # Also try another range for different lighting
        lower_skin2 = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin2 = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create masks
        skin_mask1 = cv2.inRange(frame_hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(frame_hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Only keep skin within SAM mask (person region)
        skin_mask = cv2.bitwise_and(skin_mask, sam_mask)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return skin_mask
    
    def skin_color_detection_small(self, bgr_region):
        """
        Skin detection for a small region (faster).
        
        Args:
            bgr_region: BGR image region
            
        Returns:
            np.ndarray: Binary mask of skin regions
        """
        if bgr_region.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        
        frame_hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 40, 50], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(frame_hsv, lower_skin, upper_skin)
        return skin_mask > 0
    
    
    def apply_mask(self, frame, mask):
        """
        STEP 3: Apply Mask to Frame
        Masks frame to keep only segmented region, sets background.
        
        Args:
            frame: Original frame (BGR)
            mask: Binary mask from SAM
            
        Returns:
            np.ndarray: Masked frame with specified background
        """
        masked_frame = frame.copy().astype(np.float32)
        
        # Set background based on configuration
        if self.use_blurred_background:
            # Use blurred original background (best for ViT attention)
            # This preserves spatial context while keeping background subtle
            blurred_bg = cv2.GaussianBlur(frame, (self.background_blur_size, self.background_blur_size), 0)
            blurred_bg = (blurred_bg.astype(np.float32) * self.background_brightness).astype(np.uint8)
            masked_frame[mask == 0] = blurred_bg[mask == 0]
        elif self.background_color == "gray":
            # Use gray background (128, 128, 128)
            masked_frame[mask == 0] = [128, 128, 128]
        else:
            # Default: black background (0, 0, 0)
            masked_frame[mask == 0] = [0, 0, 0]
        
        return masked_frame.astype(np.uint8)
    
    def process_frame(self, frame, remove_clothing=True):
        """
        Complete Pipeline: Input → Detection → SAM Segmentation → Clothing Removal → Brightening → Masking → Output
        
        Pipeline steps:
        1. Convert BGR to RGB
        2. Detect upper body region (MediaPipe or center-based)
        3. Segment with SAM (automatic or prompt-based)
        4. Remove clothing (keep only face, hands, exposed skin)
        5. Apply mask to frame (black background)
        
        Args:
            frame: Input frame (BGR format)
            remove_clothing: If True, remove clothing and keep only body parts
            
        Returns:
            tuple: (masked_frame, mask)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # STEP 1: Check if face_hands_only mode is enabled
        if self.face_hands_only:
            # Only extract face and hands using MediaPipe (no body segmentation needed)
            mask = self.detect_face_hands_only_mask(rgb_frame)
        else:
            # STEP 2: Detect upper body region for prompt
            bbox = self.detect_upper_body_region(rgb_frame)
            
            # STEP 3: Segment with SAM (gets entire person including clothing)
            sam_mask = self.segment_with_sam(rgb_frame, bbox=bbox)
            
            # STEP 4: Remove clothing if requested
            if remove_clothing:
                body_parts_mask = self.detect_body_parts_mask(rgb_frame, sam_mask)
                # Use body parts mask instead of full SAM mask
                mask = body_parts_mask
            else:
                mask = sam_mask
        
        # STEP 5: Apply mask
        masked_frame = self.apply_mask(frame, mask)
        
        # STEP 6: Ensure background has correct color (no noise)
        # Double-check: any pixel outside mask should have the specified background
        if self.use_blurred_background:
            # Re-apply blurred background to ensure consistency
            blurred_bg = cv2.GaussianBlur(frame, (self.background_blur_size, self.background_blur_size), 0)
            blurred_bg = (blurred_bg.astype(np.float32) * self.background_brightness).astype(np.uint8)
            masked_frame[mask == 0] = blurred_bg[mask == 0]
        elif self.background_color == "gray":
            masked_frame[mask == 0] = [128, 128, 128]
        else:
            masked_frame[mask == 0] = [0, 0, 0]
        
        # Optional final resize to fixed resolution
        if self.output_size is not None:
            if isinstance(self.output_size, (tuple, list)) and len(self.output_size) == 2:
                out_w, out_h = int(self.output_size[0]), int(self.output_size[1])
            else:
                size = int(self.output_size)
                out_w = out_h = size
            masked_frame = cv2.resize(masked_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        
        return masked_frame, mask
    
    def process_video(self, input_path, output_path, fps=None, save_mask_video=False):
        """
        Process entire video and save SAM-segmented version.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            fps: Output FPS (default: same as input)
            save_mask_video: Also save mask visualization video
        """
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps is None:
            fps = original_fps
        
        # Determine output resolution (resize if requested)
        if self.output_size is not None:
            if isinstance(self.output_size, (tuple, list)) and len(self.output_size) == 2:
                output_width = int(self.output_size[0])
                output_height = int(self.output_size[1])
            else:
                size = int(self.output_size)
                output_width = output_height = size
        else:
            output_width, output_height = width, height
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
        
        # Optional: mask visualization writer
        mask_out = None
        if save_mask_video:
            mask_output_path = str(output_path).replace('.mp4', '_mask_vis.mp4')
            mask_out = cv2.VideoWriter(mask_output_path, fourcc, fps, (output_width, output_height))
        
        # Reset temporal smoothing buffers for new video
        self.previous_mask = None
        self.previous_bbox = None
        
        # Process frames
        frame_count = 0
        print(f"📊 Processing {total_frames} frames...")
        print(f"⚡ Mode: {'Automatic (SLOW - several seconds per frame)' if self.use_automatic_generator else 'Prompt-based (FAST)'}")
        if self.temporal_smoothing > 0:
            print(f"🎯 Temporal smoothing: {self.temporal_smoothing:.1%} (reduces shakiness)")
        
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress bar description with current frame
                pbar.set_description(f"Processing frame {frame_count + 1}/{total_frames}")
                
                # Process frame
                try:
                    masked_frame, mask = self.process_frame(
                        frame, 
                        remove_clothing=self.remove_clothing
                    )
                except Exception as e:
                    print(f"\n⚠️  Error processing frame {frame_count + 1}: {e}")
                    # Continue with original frame if processing fails
                    masked_frame = frame
                    mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                
                # Write masked frame
                out.write(masked_frame)
                
                # Write mask visualization if requested
                if mask_out is not None:
                    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
                    mask_out.write(mask_vis)
                
                frame_count += 1
                pbar.update(1)
                
                # Periodic status update every 10 frames
                if frame_count % 10 == 0:
                    pbar.set_postfix({"processed": f"{frame_count}/{total_frames}"})
        
        # Release resources
        cap.release()
        out.release()
        if mask_out:
            mask_out.release()
        
        print(f"✅ Processed {frame_count} frames")
        print(f"📹 Output saved to: {output_path}")
        if save_mask_video:
            print(f"📹 Mask visualization saved to: {mask_output_path}")


def process_single_video_for_multiprocessing(video_file_path, output_dir, processor_kwargs, fps=None, save_mask_video=False):
    """
    Process a single video file - module-level function for multiprocessing.
    This function can be pickled by multiprocessing.
    
    Args:
        video_file_path: Path to input video file
        output_dir: Output directory path
        processor_kwargs: Dictionary of arguments to create SAMVideoProcessor
        fps: Output FPS (optional)
        save_mask_video: Whether to save mask visualization
        
    Returns:
        dict: Status information {'status': 'success'|'skipped'|'error', 'file': filename, ...}
    """
    video_file = Path(video_file_path)
    output_video = Path(output_dir) / f"{video_file.stem}.mp4"
    
    # Skip if output already exists
    if output_video.exists():
        return {'status': 'skipped', 'file': video_file.name}
    
    try:
        # Create processor instance for this process
        processor = SAMVideoProcessor(**processor_kwargs)
        
        # Process video
        processor.process_video(
            video_file,
            output_video,
            fps=fps,
            save_mask_video=save_mask_video
        )
        return {'status': 'success', 'file': video_file.name}
    except Exception as e:
        return {'status': 'error', 'file': video_file.name, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Process videos with SAM to segment upper body and mask background"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input video file path (required if --input-dir not specified)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Process all videos from a directory (overrides --input)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video file path (default: same as input filename)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed videos"
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to SAM model checkpoint file (required for SAM backends)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type (default: vit_h for best quality)"
    )
    parser.add_argument(
        "--segmentation-backend",
        type=str,
        default="sam",
        choices=["sam", "mediapipe"],
        help="Segmentation backend: 'sam' (default) or 'mediapipe' (fastest)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--no-automatic",
        action="store_true",
        help="Disable automatic mask generator (use prompt-based instead)"
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.95,
        help="Stability score threshold for mask filtering (default: 0.95)"
    )
    parser.add_argument(
        "--temporal-smoothing",
        type=float,
        default=0.7,
        help="Temporal smoothing factor (0.0-1.0) to reduce shakiness. Higher = more smoothing. Default: 0.5"
    )
    parser.add_argument(
        "--keep-clothing",
        action="store_true",
        help="Keep clothing in output (default: remove clothing, keep only face/hands/skin)"
    )
    parser.add_argument(
        "--face-hands-only",
        action="store_true",
        help="Only extract face and hands (no body parts). Requires MediaPipe backend. "
             "This mode is faster and ideal for sign language videos focusing on facial expressions and hand gestures."
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.0,
        help="Scale factor for bounding box size (1.0 = default, < 1.0 = smaller crop, > 1.0 = larger crop). Default: 1.0"
    )
    parser.add_argument(
        "--mediapipe-model-selection",
        type=int,
        default=1,
        choices=[0, 1],
        help="MediaPipe selfie segmentation model: 0=general, 1=landscape (better quality). Default: 1"
    )
    parser.add_argument(
        "--mediapipe-mask-threshold",
        type=float,
        default=0.5,
        help="MediaPipe mask threshold (0.0-1.0). Lower = more permissive. Default: 0.5"
    )
    parser.add_argument(
        "--mediapipe-pose-confidence",
        type=float,
        default=0.5,
        help="MediaPipe pose detection confidence threshold (0.0-1.0). Lower = more detections. Default: 0.5"
    )
    parser.add_argument(
        "--mediapipe-hands-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hands detection confidence threshold (0.0-1.0). Lower = more detections. Default: 0.5"
    )
    parser.add_argument(
        "--mediapipe-face-confidence",
        type=float,
        default=0.5,
        help="MediaPipe face detection confidence threshold (0.0-1.0). Lower = more detections. Default: 0.5"
    )
    parser.add_argument(
        "--mediapipe-face-expand",
        type=int,
        default=15,
        help="Extra pixels to expand around detected face bounding box (default: 15). Increase to include more context."
    )
    parser.add_argument(
        "--mediapipe-hand-margin",
        type=int,
        default=25,
        help="Extra pixels to expand around detected hand region (default: 25). Increase to include more context."
    )
    parser.add_argument(
        "--mediapipe-face-shrink",
        type=float,
        default=0.2,
        help="Fraction (0-1) to trim from bottom of face box to avoid neck region (default: 0.2). Set to 0 to disable."
    )
    parser.add_argument(
        "--resize-output",
        type=str,
        default=None,
        help="Resize final masked video frames. Provide a single value (e.g., 320 or 224) or WIDTHxHEIGHT (e.g., 320x256)."
    )
    parser.add_argument(
        "--save-mask-vis",
        action="store_true",
        help="Also save mask visualization video"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output video FPS (default: same as input)"
    )
    parser.add_argument(
        "--background-color",
        type=str,
        default="black",
        choices=["black", "gray", "blurred"],
        help="Background type for masked regions (default: black). "
             "Options: 'black' (pure black), 'gray' (gray), 'blurred' (blurred original background). "
             "'blurred' is recommended for better ViT attention on face/hands."
    )
    parser.add_argument(
        "--background-blur-size",
        type=int,
        default=51,
        help="Gaussian blur kernel size for blurred background (default: 51, larger = more blur)."
    )
    parser.add_argument(
        "--background-brightness",
        type=float,
        default=0.1,
        help="Brightness multiplier for blurred background (0.0-1.0, default: 0.1 = 10% of original). "
             "Lower = darker background."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = use all CPU cores, 1 = single-threaded). Default: 0"
    )
    
    args = parser.parse_args()
    
    # Check backend availability
    if args.segmentation_backend == "sam" and not SAM_AVAILABLE:
        print("❌ Error: Segment Anything Model is not installed.")
        print("   Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("   Download checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        sys.exit(1)
    
    if args.segmentation_backend == "mediapipe" and not MEDIAPIPE_AVAILABLE:
        print("❌ Error: MediaPipe is not installed.")
        print("   Install with: pip install mediapipe")
        sys.exit(1)
    
    # Check face_hands_only requirement
    if args.face_hands_only:
        if not MEDIAPIPE_AVAILABLE:
            print("❌ Error: --face-hands-only requires MediaPipe.")
            print("   Install with: pip install mediapipe")
            sys.exit(1)
        if args.segmentation_backend != "mediapipe":
            print("⚠️  Warning: --face-hands-only works best with MediaPipe backend.")
            print("   Automatically switching to MediaPipe backend.")
            args.segmentation_backend = "mediapipe"
    
    # Check checkpoint file (only required for SAM-based backends)
    checkpoint_path = None
    if args.segmentation_backend == "sam":
        if args.sam_checkpoint is None:
            print(f"❌ Error: --sam-checkpoint is required for '{args.segmentation_backend}' backend")
            print(f"   💡 Tip: Use --segmentation-backend mediapipe to skip SAM checkpoint requirement")
            sys.exit(1)
        checkpoint_path = Path(args.sam_checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Error: SAM checkpoint not found: {checkpoint_path}")
            print("   Download checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            print(f"   💡 Tip: Use --segmentation-backend mediapipe to skip SAM checkpoint requirement")
            sys.exit(1)
    elif args.segmentation_backend == "mediapipe":
        # MediaPipe doesn't need checkpoint
        checkpoint_path = None
        if args.sam_checkpoint is not None:
            print("⚠️  Warning: --sam-checkpoint is ignored when using MediaPipe backend")
    
    # Auto-detect model type from checkpoint filename if possible (only for SAM backends)
    if checkpoint_path:
        checkpoint_name = checkpoint_path.name.lower()
        detected_model_type = None
        for model_type in ["vit_h", "vit_l", "vit_b"]:
            if f"sam_{model_type}" in checkpoint_name or f"_{model_type}_" in checkpoint_name:
                detected_model_type = model_type
                break
        
        # Use detected model type if it differs from user input
        if detected_model_type and detected_model_type != args.model_type:
            print(f"⚠️  Warning: Detected model type '{detected_model_type}' from checkpoint filename,")
            print(f"   but you specified '{args.model_type}'. Using '{detected_model_type}' instead.")
            args.model_type = detected_model_type
    
    # Batch process directory mode (highest priority)
    if args.input_dir:
        if args.output_dir is None:
            print("❌ Error: --output-dir is required when using --input-dir")
            sys.exit(1)
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            print(f"❌ Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.glob(f'*{ext}'))
            video_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"⚠️  No video files found in {input_dir}")
            sys.exit(1)
        
        print("🎬 SAM-Based Video Segmentation (Batch Mode)")
        print("=" * 60)
        print(f"📹 Found {len(video_files)} video files")
        print(f"📁 Input directory: {input_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"Backend: {args.segmentation_backend}")
        if checkpoint_path:
            print(f"Model:  {args.model_type}")
        print(f"Device: {args.device}")
        
        # Determine number of workers
        num_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
        if num_workers > 1:
            print(f"🚀 Using {num_workers} parallel workers")
        else:
            print(f"🐌 Using single-threaded processing")
        
        # Parse resize option
        def parse_resize_option(opt):
            if opt is None:
                return None
            if isinstance(opt, (tuple, list)) and len(opt) == 2:
                return (int(opt[0]), int(opt[1]))
            try:
                if isinstance(opt, int):
                    return (opt, opt)
                if isinstance(opt, str):
                    if "x" in opt.lower():
                        w_str, h_str = opt.lower().split("x")
                        return (int(w_str), int(h_str))
                    return (int(opt), int(opt))
            except ValueError:
                print(f"⚠️  Invalid --resize-output value '{opt}', ignoring resize.")
            return None
        
        resize_output = parse_resize_option(args.resize_output)
        
        # Prepare processor kwargs
        processor_kwargs = {
            'sam_checkpoint': str(checkpoint_path) if checkpoint_path else None,
            'model_type': args.model_type,
            'device': args.device,
            'use_automatic_generator': not args.no_automatic,
            'stability_threshold': args.stability_threshold,
            'temporal_smoothing': args.temporal_smoothing,
            'remove_clothing': not args.keep_clothing,
            'segmentation_backend': args.segmentation_backend,
            'crop_scale': args.crop_scale,
            'mediapipe_model_selection': args.mediapipe_model_selection,
            'mediapipe_mask_threshold': args.mediapipe_mask_threshold,
            'mediapipe_pose_confidence': args.mediapipe_pose_confidence,
            'mediapipe_hands_confidence': args.mediapipe_hands_confidence,
            'mediapipe_face_confidence': args.mediapipe_face_confidence,
            'mediapipe_face_expand': args.mediapipe_face_expand,
            'mediapipe_hand_margin': args.mediapipe_hand_margin,
            'mediapipe_face_shrink': args.mediapipe_face_shrink,
            'output_size': resize_output,
            'background_color': args.background_color,
            'background_blur_size': args.background_blur_size,
            'background_brightness': args.background_brightness,
            'face_hands_only': args.face_hands_only
        }
        
        print("-" * 60)
        
        # Process videos
        success_count = 0
        skipped_count = 0
        error_count = 0
        
        if num_workers > 1:
            # Multiprocessing mode
            video_paths = [str(vf) for vf in video_files]
            process_func = partial(
                process_single_video_for_multiprocessing,
                output_dir=str(output_dir),
                processor_kwargs=processor_kwargs,
                fps=args.fps,
                save_mask_video=args.save_mask_vis
            )
            
            with multiprocessing.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, video_paths),
                    total=len(video_paths),
                    desc="Processing videos"
                ))
        else:
            # Single-threaded mode
            results = []
            for video_file in tqdm(video_files, desc="Processing videos"):
                result = process_single_video_for_multiprocessing(
                    str(video_file),
                    str(output_dir),
                    processor_kwargs,
                    fps=args.fps,
                    save_mask_video=args.save_mask_vis
                )
                results.append(result)
        
        # Count results
        for result in results:
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skipped':
                skipped_count += 1
            elif result['status'] == 'error':
                error_count += 1
                print(f"❌ Error processing {result['file']}: {result.get('error', 'Unknown error')}")
        
        print("-" * 60)
        print(f"✅ Successfully processed: {success_count} videos")
        if skipped_count > 0:
            print(f"⏭️  Skipped (already exist): {skipped_count} videos")
        if error_count > 0:
            print(f"❌ Errors: {error_count} videos")
        return
    
    # Single video mode
    if args.input is None:
        print("❌ Error: --input or --input-dir is required")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input video not found: {input_path}")
        sys.exit(1)
    
    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.mp4"
    else:
        output_path = input_path.parent / f"{input_path.stem}.mp4"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🎬 SAM-Based Video Segmentation")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Backend: {args.segmentation_backend}")
    if checkpoint_path:
        print(f"Model:  {args.model_type}")
    print(f"Device: {args.device}")
    if args.segmentation_backend == "sam":
        print(f"Mode:   {'Automatic' if not args.no_automatic else 'Prompt-based'}")
    if args.face_hands_only:
        print(f"Mode:   Face and Hands Only (MediaPipe)")
    else:
        print(f"Clothing: {'Keep' if args.keep_clothing else 'Remove (face/hands/skin only)'}")
    # Parse resize option
    def parse_resize_option(opt):
        if opt is None:
            return None
        if isinstance(opt, (tuple, list)) and len(opt) == 2:
            return (int(opt[0]), int(opt[1]))
        try:
            if isinstance(opt, int):
                return (opt, opt)
            if isinstance(opt, str):
                if "x" in opt.lower():
                    w_str, h_str = opt.lower().split("x")
                    return (int(w_str), int(h_str))
                return (int(opt), int(opt))
        except ValueError:
            print(f"⚠️  Invalid --resize-output value '{opt}', ignoring resize.")
        return None
    
    resize_output = parse_resize_option(args.resize_output)
    
    if args.crop_scale != 1.0:
        print(f"Crop Scale: {args.crop_scale:.2f} ({'smaller' if args.crop_scale < 1.0 else 'larger'} crop)")
    print(f"Background Color: {args.background_color}")
    if args.background_color == "blurred":
        print(f"  Blur Size: {args.background_blur_size}, Brightness: {args.background_brightness:.2f} (recommended for ViT attention)")
    elif args.background_color == "gray":
        print(f"  (may help with ViT attention)")
    if args.segmentation_backend == "mediapipe":
        print(f"MediaPipe Model: {args.mediapipe_model_selection} ({'general' if args.mediapipe_model_selection == 0 else 'landscape'})")
        print(f"MediaPipe Mask Threshold: {args.mediapipe_mask_threshold:.2f}")
        print(f"MediaPipe Confidences: pose={args.mediapipe_pose_confidence:.2f}, hands={args.mediapipe_hands_confidence:.2f}, face={args.mediapipe_face_confidence:.2f}")
        print(f"MediaPipe Face Expand: {args.mediapipe_face_expand}px")
        print(f"MediaPipe Hand Margin: {args.mediapipe_hand_margin}px")
        print(f"MediaPipe Face Shrink: {args.mediapipe_face_shrink:.2f}")
    if resize_output:
        print(f"Resize Output: {resize_output[0]}x{resize_output[1]}")
    print("-" * 60)
    
    # Initialize processor
    try:
        processor = SAMVideoProcessor(
            sam_checkpoint=str(checkpoint_path) if checkpoint_path else None,
            model_type=args.model_type,
            device=args.device,
            use_automatic_generator=not args.no_automatic,
            stability_threshold=args.stability_threshold,
            temporal_smoothing=args.temporal_smoothing,
            remove_clothing=not args.keep_clothing,
            segmentation_backend=args.segmentation_backend,
            crop_scale=args.crop_scale,
            mediapipe_model_selection=args.mediapipe_model_selection,
            mediapipe_mask_threshold=args.mediapipe_mask_threshold,
            mediapipe_pose_confidence=args.mediapipe_pose_confidence,
            mediapipe_hands_confidence=args.mediapipe_hands_confidence,
            mediapipe_face_confidence=args.mediapipe_face_confidence,
            mediapipe_face_expand=args.mediapipe_face_expand,
            mediapipe_hand_margin=args.mediapipe_hand_margin,
            mediapipe_face_shrink=args.mediapipe_face_shrink,
            output_size=resize_output,
            background_color=args.background_color,
            background_blur_size=args.background_blur_size,
            background_brightness=args.background_brightness,
            face_hands_only=args.face_hands_only
        )
        
        # Process video
        processor.process_video(
            input_path,
            output_path,
            fps=args.fps,
            save_mask_video=args.save_mask_vis
        )
        
        print("✅ Processing complete!")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

