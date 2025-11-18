#!/usr/bin/env python3
"""
Generate per-frame binary ROI masks (hands + face) using MediaPipe and
store them as compressed numpy arrays.

The resulting mask tensor M_fg can be consumed during training to build
foreground/background clips on-the-fly for FBCF.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "MediaPipe is required for this script. Install via `pip install mediapipe`."
    ) from exc


def parse_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    try:
        if "x" in value.lower():
            w_str, h_str = value.lower().split("x")
            return int(w_str), int(h_str)
        size = int(value)
        return size, size
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid size specification '{value}'. Use int or WxH."
        ) from exc


class FaceHandMaskGenerator:
    """Create binary masks covering face and hands using MediaPipe detectors."""

    def __init__(
        self,
        face_expand: int = 15,
        face_shrink: float = 0.2,
        hand_margin: int = 25,
        face_conf: float = 0.5,
        hand_conf: float = 0.5,
        blur_kernel: int = 7,
        morph_kernel: int = 5,
    ):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=face_conf
        )
        self.hands_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=hand_conf,
            min_tracking_confidence=hand_conf,
        )
        self.face_expand = face_expand
        self.face_shrink = face_shrink
        self.hand_margin = hand_margin
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.morph_kernel = morph_kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )

    def _face_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        results = self.face_detector.process(frame_rgb)
        if not results.detections:
            return mask

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Optional shrink from bottom to avoid torso
            if self.face_shrink > 0:
                height = max(1, int(height * (1 - self.face_shrink)))

            expand = self.face_expand
            x = max(0, x - expand)
            y = max(0, y - expand)
            width = min(w - x, width + 2 * expand)
            height = min(h - y, height + 2 * expand)
            mask[y : y + height, x : x + width] = 1
        return mask

    def _hands_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        results = self.hands_detector.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return mask

        for landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            if not xs or not ys:
                continue
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            margin = self.hand_margin
            x1 = max(0, int(x_min) - margin)
            y1 = max(0, int(y_min) - margin)
            x2 = min(w, int(x_max) + margin)
            y2 = min(h, int(y_max) + margin)
            mask[y1:y2, x1:x2] = 1
        return mask

    def generate_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = self._face_mask(frame_rgb)
        mask = np.maximum(mask, self._hands_mask(frame_rgb))

        # Morphological cleanup and blur for smoother boundaries
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.GaussianBlur(
            mask.astype(np.float32), (self.blur_kernel, self.blur_kernel), 0
        )
        return (mask > 0.5).astype(np.uint8)


def save_masks_npz(
    video_path: Path,
    output_dir: Path,
    generator: FaceHandMaskGenerator,
    resize_to: Optional[Tuple[int, int]] = None,
    preview: bool = False,
) -> Tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resized_dims = resize_to if resize_to else (width, height)
    masks: List[np.ndarray] = []
    preview_writer = None
    if preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_path = output_dir / f"{video_path.stem}_mask_preview.mp4"
        preview_writer = cv2.VideoWriter(
            str(preview_path), fourcc, fps, resized_dims, isColor=True
        )

    frame_idx = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=video_path.stem) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            mask = generator.generate_mask(frame)
            if resize_to:
                frame = cv2.resize(frame, resized_dims, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, resized_dims, interpolation=cv2.INTER_NEAREST)
            masks.append(mask.astype(np.uint8))

            if preview_writer is not None:
                overlay = frame.copy()
                overlay[mask > 0] = (0, 255, 0)
                blend = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                preview_writer.write(blend)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    if preview_writer is not None:
        preview_writer.release()

    if not masks:
        raise RuntimeError(f"No frames read from {video_path}")

    mask_array = np.stack(masks, axis=0).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.npz"
    np.savez_compressed(
        output_path,
        mask=mask_array,
        fps=fps,
        height=resized_dims[1],
        width=resized_dims[0],
    )
    return mask_array.shape[0], resized_dims[0], resized_dims[1]


def collect_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    video_exts = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    videos = []
    for ext in video_exts:
        videos.extend(input_path.rglob(f"*{ext}"))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description="Save MediaPipe ROI masks per frame.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store mask npz files.",
    )
    parser.add_argument(
        "--resize",
        type=parse_size,
        default=None,
        help="Resize masks (and optional preview) to WIDTHxHEIGHT before saving.",
    )
    parser.add_argument("--face-expand", type=int, default=15)
    parser.add_argument("--face-shrink", type=float, default=0.2)
    parser.add_argument("--hand-margin", type=int, default=25)
    parser.add_argument("--face-conf", type=float, default=0.5)
    parser.add_argument("--hand-conf", type=float, default=0.5)
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=7,
        help="Kernel size for Gaussian blur (odd number).",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="Kernel size for morphological operations (odd number).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also save an mp4 preview overlay for debugging.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    generator = FaceHandMaskGenerator(
        face_expand=args.face_expand,
        face_shrink=args.face_shrink,
        hand_margin=args.hand_margin,
        face_conf=args.face_conf,
        hand_conf=args.hand_conf,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
    )

    videos = collect_videos(input_path)
    if not videos:
        raise RuntimeError(f"No video files found under {input_path}")

    print(f"Found {len(videos)} videos. Saving masks to {output_dir.resolve()}")
    base_dir = input_path.parent if input_path.is_file() else input_path
    for video_path in videos:
        rel_dir = video_path.parent.relative_to(base_dir)
        target_dir = output_dir / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            num_frames, out_w, out_h = save_masks_npz(
                video_path,
                target_dir,
                generator,
                resize_to=args.resize,
                preview=args.preview,
            )
            print(
                f"✅ Saved mask for {video_path.name} "
                f"({num_frames} frames @ {out_w}x{out_h})"
            )
        except Exception as exc:
            print(f"❌ Failed on {video_path}: {exc}")


if __name__ == "__main__":
    main()


