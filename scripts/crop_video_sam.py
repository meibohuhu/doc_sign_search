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
    
    def __init__(self, sam_checkpoint, model_type="vit_h", device="cuda", 
                 use_automatic_generator=True, stability_threshold=0.95):
        """
        Initialize SAM video processor.
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            device: Device to use ("cuda" or "cpu")
            use_automatic_generator: Use automatic mask generator (better quality)
            stability_threshold: Stability score threshold for mask filtering
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.use_automatic_generator = use_automatic_generator
        self.stability_threshold = stability_threshold
        
        if not SAM_AVAILABLE:
            raise ImportError(
                "Segment Anything Model is required.\n"
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                "Download checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        
        print(f"🚀 Loading SAM model ({model_type}) on {self.device}...")
        
        # Load SAM model
        try:
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=self.device)
            self.sam.eval()
            
            if use_automatic_generator:
                # Use automatic mask generator for better quality
                self.mask_generator = SamAutomaticMaskGenerator(
                    self.sam,
                    points_per_side=32,
                    pred_iou_thresh=0.9,
                    stability_score_thresh=stability_threshold,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
            else:
                # Use predictor for prompt-based segmentation
                self.predictor = SamPredictor(self.sam)
            
            print("✅ SAM model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading SAM model: {e}")
            raise
        
        # Initialize MediaPipe for upper body detection (optional)
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            self.use_mediapipe = True
        else:
            self.use_mediapipe = False
    
    def detect_upper_body_region(self, rgb_frame):
        """
        STEP 1: Detect Upper Body Region
        Detects upper body bounding box for SAM prompt.
        
        Args:
            rgb_frame: RGB frame from video
            
        Returns:
            tuple: (x, y, width, height) bounding box or None
        """
        h, w = rgb_frame.shape[:2]
        
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
                    
                    # Add margin
                    margin_x = int((x_max - x_min) * 0.1)
                    margin_y = int((y_max - y_min) * 0.1)
                    
                    x = max(0, x_min - margin_x)
                    y = max(0, y_min - margin_y)
                    width = min(w - x, x_max - x_min + 2 * margin_x)
                    height = min(h - y, y_max - y_min + 2 * margin_y)
                    
                    return x, y, width, height
        
        # Fallback: use center region (assumes signer is centered)
        center_x, center_y = w // 2, h // 2
        bbox_width = int(w * 0.6)  # 60% of width
        bbox_height = int(h * 0.7)  # 70% of height
        
        x = center_x - bbox_width // 2
        y = center_y - bbox_height // 2
        
        return x, y, bbox_width, bbox_height
    
    def segment_with_sam(self, rgb_frame, bbox=None, point_prompt=None):
        """
        STEP 2: SAM Segmentation Pipeline
        Segments the upper body region using SAM.
        
        Args:
            rgb_frame: RGB frame from video
            bbox: Optional bounding box (x, y, width, height) as prompt
            point_prompt: Optional center point as prompt
            
        Returns:
            np.ndarray: Binary mask (255 for foreground, 0 for background)
        """
        h, w = rgb_frame.shape[:2]
        
        if self.use_automatic_generator:
            # Automatic mask generation (better quality, slower)
            masks = self.mask_generator.generate(rgb_frame)
            
            if not masks:
                # Fallback: return empty mask
                return np.zeros((h, w), dtype=np.uint8)
            
            # Select the best mask (largest area, highest stability)
            # Filter by stability score
            valid_masks = [m for m in masks if m.get('stability_score', 0) >= self.stability_threshold]
            
            if not valid_masks:
                # Use all masks if none meet threshold
                valid_masks = masks
            
            # Find the largest mask (likely the person)
            best_mask = max(valid_masks, key=lambda m: m['area'])
            
            # Convert to binary mask
            mask = best_mask['segmentation'].astype(np.uint8) * 255
            
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
            
            # Select the mask with highest score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx].astype(np.uint8) * 255
        
        # Post-process mask for smoother edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def apply_mask(self, frame, mask):
        """
        STEP 3: Apply Mask to Frame
        Masks frame to keep only segmented region, black out background.
        
        Args:
            frame: Original frame (BGR)
            mask: Binary mask from SAM
            
        Returns:
            np.ndarray: Masked frame with black background
        """
        masked_frame = frame.copy()
        masked_frame[mask == 0] = 0
        return masked_frame
    
    def process_frame(self, frame):
        """
        Complete Pipeline: Input → Detection → SAM Segmentation → Masking → Output
        
        Pipeline steps:
        1. Convert BGR to RGB
        2. Detect upper body region (MediaPipe or center-based)
        3. Segment with SAM (automatic or prompt-based)
        4. Apply mask to frame (black background)
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (masked_frame, mask)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # STEP 1: Detect upper body region for prompt
        bbox = self.detect_upper_body_region(rgb_frame)
        
        # STEP 2: Segment with SAM
        mask = self.segment_with_sam(rgb_frame, bbox=bbox)
        
        # STEP 3: Apply mask
        masked_frame = self.apply_mask(frame, mask)
        
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
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Optional: mask visualization writer
        mask_out = None
        if save_mask_video:
            mask_output_path = str(output_path).replace('.mp4', '_mask_vis.mp4')
            mask_out = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                masked_frame, mask = self.process_frame(frame)
                
                # Write masked frame
                out.write(masked_frame)
                
                # Write mask visualization if requested
                if mask_out is not None:
                    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
                    mask_out.write(mask_vis)
                
                frame_count += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        if mask_out:
            mask_out.release()
        
        print(f"✅ Processed {frame_count} frames")
        print(f"📹 Output saved to: {output_path}")
        if save_mask_video:
            print(f"📹 Mask visualization saved to: {mask_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process videos with SAM to segment upper body and mask background"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video file path (default: input_sam_masked.mp4)"
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
        required=True,
        help="Path to SAM model checkpoint file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type (default: vit_h for best quality)"
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
    
    args = parser.parse_args()
    
    # Check SAM availability
    if not SAM_AVAILABLE:
        print("❌ Error: Segment Anything Model is not installed.")
        print("   Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("   Download checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        sys.exit(1)
    
    # Check checkpoint file
    checkpoint_path = Path(args.sam_checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: SAM checkpoint not found: {checkpoint_path}")
        print("   Download checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        sys.exit(1)
    
    # Resolve input path
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
        output_path = output_dir / f"{input_path.stem}_sam_masked.mp4"
    else:
        output_path = input_path.parent / f"{input_path.stem}_sam_masked.mp4"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🎬 SAM-Based Video Segmentation")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Mode:   {'Automatic' if not args.no_automatic else 'Prompt-based'}")
    print("-" * 60)
    
    # Initialize processor
    try:
        processor = SAMVideoProcessor(
            sam_checkpoint=str(checkpoint_path),
            model_type=args.model_type,
            device=args.device,
            use_automatic_generator=not args.no_automatic,
            stability_threshold=args.stability_threshold
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

