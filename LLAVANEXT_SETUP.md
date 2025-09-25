# LLaVA-NeXT Video Setup Instructions

## Overview

This setup uses the official LLaVA-NeXT Video model for video understanding tasks, specifically person counting in videos. The implementation follows the patterns from the [LLaVA-NeXT documentation](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_Video_1003.md).

## Installation

The required packages are already installed in your `llava` environment:

- `torch`
- `transformers`
- `decord` (for video processing)
- `PIL` (Pillow)
- `cv2` (opencv-python)
- `numpy`
- `tqdm`

## Model Options

### Default Model (Recommended):
```bash
lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only
```

### Alternative Models:
```bash
lmms-lab/LLaVA-Video-7B-Qwen2
lmms-lab/LLaVA-Video-7B-Qwen2.5-Video-Only
```

## Usage

### Run with all samples:
```bash
bash scripts/eval/run_llavanext_test.sh
```

### Run with limited samples:
```bash
bash scripts/eval/run_llavanext_test.sh 5
```

### Run directly with Python:
```bash
python -m playground.demo.llavanext_metrics \
    --model-path lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos.json \
    --video-folder "/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/" \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --max-samples 1 \
    --add_time_instruction
```

## Key Features

### Video Processing:
- **Frame Sampling**: Uses uniform sampling with configurable frame counts
- **Time Instructions**: Includes temporal information in prompts
- **Memory Optimization**: Adjusts frame count based on video size
- **Official Pattern**: Follows LLaVA-NeXT video processing conventions

### Frame Count Strategy:
- **Small videos (<5MB)**: 32 frames
- **Medium videos (5-20MB)**: 16 frames  
- **Large videos (>20MB)**: 8 frames

### Generation Settings:
- **Temperature**: 0.2 (deterministic for counting tasks)
- **Max Tokens**: 512
- **Sampling**: Deterministic by default
- **Modalities**: ["video"] for proper video processing

## Output

Results will be saved to:
- `/local1/mhu/LLaVANeXT_RC/new_outputs/llavanext_results.json` - Model outputs
- `/local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics.json` - Evaluation metrics

## Technical Details

### Video Loading:
Based on the official LLaVA-NeXT pattern:
```python
def load_video(video_path, max_frames_num, fps=1, force_sample=True):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    # Uniform frame sampling
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time_str, video_time
```

### Model Loading:
```python
tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path, args.model_base, args.model_name, 
    device_map=device_map, attn_implementation="sdpa", 
    multimodal=True
)
```

### Generation:
```python
cont = model.generate(
    input_ids,
    images=video_list,
    modalities=["video"],
    do_sample=args.do_sample,
    temperature=args.temperature,
    max_new_tokens=args.max_new_tokens,
    use_cache=True
)
```

## Notes

- The script automatically detects conversation templates based on model type
- Supports both video and image inputs
- Includes comprehensive error handling and fallback mechanisms
- Uses official LLaVA-NeXT video processing patterns from the GitHub repository
- Optimized for person counting tasks with appropriate prompts and settings
