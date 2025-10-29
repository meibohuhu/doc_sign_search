# Video Compression Guide for Training/Inference

**CRITICAL:** Videos must be compressed to match training configuration!

## Why Compress to 320×320?

Your training uses:
```bash
--video_min_pixels 102400  # 320 × 320
--video_max_pixels 102400  # 320 × 320
```

Compressing videos to 320×320 beforehand ensures:
- ✅ Consistent input during training/inference
- ✅ Faster processing (no runtime resizing)
- ✅ Reduced memory usage
- ✅ Better evaluation accuracy

## Updated Script: `stable_compress_how2sign_videos.py`

### New Defaults (Match Training):
- ✅ Resolution: **320×320** (was 224×224)
- ✅ FPS: **18** (How2Sign default, was 24)
- ✅ Output dir: `segmented_clips_stable_320x320/`

## Usage Examples

### 1. How2Sign Videos (18 fps)
```bash
python scripts/stable_compress_how2sign_videos.py \
    --input-dir /path/to/how2sign/videos \
    --output-dir /path/to/output_320x320 \
    --size 320 \
    --fps 18
```

### 2. DailyMoth Videos (12 fps)
```bash
python scripts/stable_compress_how2sign_videos.py \
    --input-dir /path/to/dailymoth/videos \
    --output-dir /path/to/output_320x320 \
    --size 320 \
    --fps 12
```

### 3. Test on 10 Videos First
```bash
python scripts/stable_compress_how2sign_videos.py \
    --test-samples 10 \
    --size 320 \
    --fps 18
```

### 4. Overwrite Existing Files
```bash
python scripts/stable_compress_how2sign_videos.py \
    --overwrite \
    --size 320 \
    --fps 18
```

## Default Behavior (No Arguments)

```bash
python scripts/stable_compress_how2sign_videos.py
```

**What it does:**
- Input: `/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips`
- Output: `/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_320x320`
- Resolution: 320×320
- FPS: 18
- Method: Stable crop (no shaking)

## Configuration Matrix

| Dataset   | Resolution | FPS | Command |
|-----------|-----------|-----|---------|
| How2Sign  | 320×320   | 18  | `--size 320 --fps 18` (default) |
| DailyMoth | 320×320   | 12  | `--size 320 --fps 12` |

## What the Script Does

### 1. Stable Crop Region Detection
- Samples 10 frames throughout video
- Detects face positions
- Calculates average stable region
- Expands generously for full body (factor 5.2x)
- **Result:** No shaking between frames! ✅

### 2. Square Cropping
- Centers crop on detected region
- Makes square (min of width/height)
- Preserves sign language gestures

### 3. Resizing
- Resizes to target (320×320)
- Uses `cv2.INTER_AREA` for quality
- Maintains aspect ratio

### 4. FPS Adjustment
- Resamples to target FPS (18 or 12)
- Consistent temporal resolution

## Output

```
✅ Stable compression: video_001.mp4
   Resolution: 1280x720 → 320x320
   FPS: 29.97 → 18.00
   Size: 45.23MB → 5.67MB (8.0x compression)
   Method: Stable crop region (no shaking)
```

## Important Notes

### ⚠️ Match Training Configuration!

**Before Training:**
```bash
# Compress videos to 320×320
python scripts/stable_compress_how2sign_videos.py --size 320 --fps 18
```

**Training Script:**
```bash
--video_min_pixels $((320 * 320))
--video_max_pixels $((320 * 320))
--fps 18
```

**Inference/Evaluation:**
```bash
--min-pixels 102400  # 320×320
--max-pixels 102400  # 320×320
--video-fps 18
```

### ✅ Benefits of Pre-compression

1. **Training:**
   - Faster data loading
   - Consistent input sizes
   - Reduced I/O bottleneck

2. **Inference:**
   - No runtime resizing needed
   - Exact same preprocessing
   - Better evaluation accuracy

3. **Storage:**
   - 5-10x compression
   - Easier to manage/backup

## Full Example Workflow

```bash
# Step 1: Compress test videos to 320×320 @ 18fps
python scripts/stable_compress_how2sign_videos.py \
    --input-dir /home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips \
    --output-dir /home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_320x320 \
    --size 320 \
    --fps 18

# Step 2: Update evaluation script to use compressed videos
# In run_qwen2vl_evaluation_how2sign.sh:
VIDEO_FOLDER="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_320x320"

# Step 3: Run evaluation with matching settings
--video-fps 18 \
--min-pixels 102400 \
--max-pixels 102400
```

## Troubleshooting

### Issue: "Video shaking/jittering"
**Solution:** Use stable compression (this script!) - it detects one crop region for entire video

### Issue: "Different results between training and inference"
**Solution:** Ensure compression settings match:
- Same resolution (320×320)
- Same FPS (18 for How2Sign, 12 for DailyMoth)
- Same processing pipeline

### Issue: "Output videos too large"
**Solution:** Already optimized! 320×320 gives 5-10x compression vs original

### Issue: "Sign language gestures cut off"
**Solution:** Script uses generous expansion (5.2x face size, 1.8x height multiplier) - captures full torso and arms

## Performance

- **Speed:** ~2-5 videos/second (depends on video length)
- **Compression:** 5-10x size reduction
- **Quality:** High quality with cv2.INTER_AREA
- **Stability:** No frame-to-frame jitter ✅

---

**Last Updated:** 2025-10-22
**Script:** `stable_compress_how2sign_videos.py`
**Training Config:** 320×320, FPS 18 (How2Sign) / 12 (DailyMoth)



