# Qwen2VL Evaluation on A6000

This directory contains scripts for running Qwen2VL inference on A6000 GPU with your custom paths.

## Files

### Fine-tuned Model Evaluation
- `run_qwen2vl_evaluation_a6000.sh`: Shell script to run evaluation with fine-tuned checkpoint
- `qwen2vl_evaluation_a6000.py`: Python script that performs inference with fine-tuned model

### Base Model Evaluation (No Fine-tuning)
- `run_qwen2vl_base_evaluation_a6000.sh`: Shell script to run evaluation with pretrained base model
- `qwen2vl_base_evaluation_a6000.py`: Python script that performs inference with base model only

## Configuration

The scripts are configured with the following paths:

- **Checkpoint**: `/local1/mhu/sign_language_llm/how2sign/checkpoints/qwen2vl_how2sign_4xa100_filtered_32batchsize_fast`
- **Model Base**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Video Folder**: `/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/segmented_clips`
- **Question File**: `/local1/mhu/sign_language_llm/vanshika/asl_test/segmented_videos.json`
- **Output Directory**: `/local1/mhu/sign_language_llm/outputs/a6000_evaluation`

## Prerequisites

Make sure you have the required environment:
1. CUDA-enabled GPU (A6000 or similar)
2. Conda environment `qwen25_vl_sign` with required packages (transformers, peft, qwen-vl-utils, etc.)
   - The scripts automatically activate this environment from `/home/ztao/anaconda3/envs/qwen25_vl_sign`
3. Access to the Qwen2.5-VL model

## Usage

### Fine-tuned Model Evaluation

#### Option 1: Run the Shell Script (Recommended)

```bash
cd /local1/mhu/sign_language_llm
bash scripts/local_eval/run_qwen2vl_evaluation_a6000.sh
```

#### Option 2: Run the Python Script Directly

```bash
cd /local1/mhu/sign_language_llm

python scripts/local_eval/qwen2vl_evaluation_a6000.py \
    --checkpoint-path "/local1/mhu/sign_language_llm/how2sign/checkpoints/qwen2vl_how2sign_4xa100_filtered_32batchsize_fast" \
    --model-base "Qwen/Qwen2.5-VL-3B-Instruct" \
    --video-folder "/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/segmented_clips" \
    --question-file "/local1/mhu/sign_language_llm/vanshika/asl_test/segmented_videos.json" \
    --out-dir "/local1/mhu/sign_language_llm/outputs/a6000_evaluation" \
    --max-samples 1000 \
    --enable-evaluation \
    --video-fps 18
```

### Base Model Evaluation (No Fine-tuning)

To evaluate the pretrained Qwen2.5-VL-3B-Instruct model without any fine-tuning (baseline):

#### Option 1: Run the Shell Script (Recommended)

```bash
cd /local1/mhu/sign_language_llm
bash scripts/local_eval/run_qwen2vl_base_evaluation_a6000.sh
```

#### Option 2: Run the Python Script Directly

```bash
cd /local1/mhu/sign_language_llm

python scripts/local_eval/qwen2vl_base_evaluation_a6000.py \
    --model-base "Qwen/Qwen2.5-VL-3B-Instruct" \
    --video-folder "/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/segmented_clips" \
    --question-file "/local1/mhu/sign_language_llm/vanshika/asl_test/segmented_videos.json" \
    --out-dir "/local1/mhu/sign_language_llm/outputs/a6000_base_evaluation" \
    --max-samples 1000 \
    --enable-evaluation \
    --video-fps 18
```

## Key Parameters

- `--max-samples`: Limit the number of samples to process (use `None` or remove to process all)
- `--video-fps`: FPS for video processing (default: 12, set to 18 to match training)
- `--enable-evaluation`: Enable automatic evaluation metrics calculation
- `--freeze-vision-tower`: Add this flag if vision tower was frozen during training
- `--save-frames`: Extract and save video frames (useful for debugging)
- `--max-new-tokens`: Maximum tokens to generate (default: 512)

## Output

### Fine-tuned Model Output
- `qwen2vl_a6000_results_YYYYMMDD_HHMMSS.json`: Model predictions and ground truth
- `evaluation_metrics_YYYYMMDD_HHMMSS.json`: Evaluation metrics (if --enable-evaluation is used)

### Base Model Output
- `qwen2vl_base_a6000_results_YYYYMMDD_HHMMSS.json`: Base model predictions and ground truth
- `base_evaluation_metrics_YYYYMMDD_HHMMSS.json`: Base model evaluation metrics (if --enable-evaluation is used)

## Adjusting for Different Sample Sizes

To test with fewer samples (faster):
```bash
# Edit the shell script and change:
--max-samples 1000    # Change to desired number (e.g., 10, 100, etc.)
```

To process all samples:
```bash
# Remove or comment out the --max-samples line in the shell script
```

## GPU Memory Optimization

The script includes several optimizations for A6000:
- Mixed precision (FP16) inference
- Automatic GPU cache clearing between samples
- Memory-efficient processing settings

If you encounter out-of-memory errors:
1. Reduce `--max-samples` to process in smaller batches
2. Lower `--video-fps` to reduce frames per video
3. Adjust `--max-new-tokens` if needed

## Troubleshooting

### Video files not found
Make sure the video folder path is correct and contains the video files referenced in the question file.

### CUDA errors
The script handles CUDA errors gracefully and will skip problematic videos while continuing with the rest.

### Import errors
Ensure all required packages are installed and the Python paths are correctly set in the script.

## Notes

### Prompts
- **Fine-tuned model**: Uses the prompt "Translate the American Sign Language in this video to English."
- **Base model**: Uses a more detailed prompt to guide the pretrained model better: "Translate the ASL signs in this video to English text. Provide only the English translation without describing the person, gestures, or video content. Answer in one sentence only. If you cannot determine the meaning, RESPOND with nothing."

### Fine-tuning Considerations
- Vision tower freezing: If your checkpoint was trained with a frozen vision tower, add the `--freeze-vision-tower` flag
- The fine-tuned evaluation loads LoRA weights and non-LoRA trainable weights (vision tower + merger)
- The base evaluation only loads the pretrained model without any checkpoints

### Hardware
- The scripts are optimized for A6000 GPU (48GB VRAM) but should work on other GPUs with adjustments
- Both evaluations use mixed precision (FP16) for memory efficiency

