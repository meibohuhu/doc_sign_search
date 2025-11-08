# Quick Start Guide - A6000 Inference

## 📋 Summary

You now have **two evaluation setups** for running Qwen2VL inference on your A6000:

1. **Fine-tuned Model Evaluation** - Tests your trained checkpoint
2. **Base Model Evaluation** - Tests the pretrained model (baseline comparison)

## 🚀 Quick Commands

### Run Fine-tuned Model (with checkpoint)
```bash
cd /local1/mhu/sign_language_llm
bash scripts/local_eval/run_qwen2vl_evaluation_a6000.sh
```

### Run Base Model (no fine-tuning, baseline)
```bash
cd /local1/mhu/sign_language_llm
bash scripts/local_eval/run_qwen2vl_base_evaluation_a6000.sh
```

## 📁 File Structure

```
scripts/local_eval/
├── run_qwen2vl_evaluation_a6000.sh          # Fine-tuned model shell script
├── qwen2vl_evaluation_a6000.py              # Fine-tuned model Python script
├── run_qwen2vl_base_evaluation_a6000.sh     # Base model shell script  
├── qwen2vl_base_evaluation_a6000.py         # Base model Python script
├── README_A6000.md                          # Detailed documentation
└── QUICK_START.md                           # This file
```

## ⚙️ Current Configuration

### Paths (both scripts)
- **Video Folder**: `/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/segmented_clips`
- **Question File**: `/local1/mhu/sign_language_llm/vanshika/asl_test/segmented_videos.json`

### Fine-tuned Model Only
- **Checkpoint**: `/local1/mhu/sign_language_llm/how2sign/checkpoints/qwen2vl_how2sign_4xa100_filtered_32batchsize_fast`
  - ⚠️ **NOTE**: This directory is currently empty - you need to add checkpoint files here

### Model Settings
- **Base Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Video FPS**: 18
- **Max Samples**: 1000 (adjustable)

## 🔧 Common Modifications

### Test with fewer samples
Edit the shell script and change:
```bash
--max-samples 1000    # Change to 10, 100, etc. for faster testing
```

### Process all samples
Remove or comment out the `--max-samples` line in the shell script.

### Change video FPS
Edit the shell script:
```bash
--video-fps 18    # Change to 12, 24, etc.
```

## 📊 Output Locations

### Fine-tuned Model
- Results: `/local1/mhu/sign_language_llm/outputs/a6000_evaluation/`
- Files: `qwen2vl_a6000_results_*.json`, `evaluation_metrics_*.json`

### Base Model
- Results: `/local1/mhu/sign_language_llm/outputs/a6000_base_evaluation/`
- Files: `qwen2vl_base_a6000_results_*.json`, `base_evaluation_metrics_*.json`

## ⚠️ Before Running

1. **Checkpoint Files**: Make sure your checkpoint files are in the correct location (fine-tuned model only)
2. **Conda Environment**: The scripts automatically activate the `qwen25_vl_sign` conda environment
   - Environment path: `/home/ztao/anaconda3/envs/qwen25_vl_sign`
   - No need to manually activate - the scripts handle this
3. **GPU**: Ensure A6000 is available and not in use

## 🆘 Troubleshooting

### "Checkpoint path not found"
The checkpoint directory exists but is empty. Copy your checkpoint files to:
`/local1/mhu/sign_language_llm/how2sign/checkpoints/qwen2vl_how2sign_4xa100_filtered_32batchsize_fast/`

### "Video not found"
Check that the video folder path is correct and contains `.mp4` files.

### Out of memory
- Reduce `--max-samples` to process fewer videos
- Lower `--video-fps` to use fewer frames per video

### CUDA errors
The scripts handle CUDA errors gracefully and will skip problematic videos while continuing.

## 📖 More Information

See `README_A6000.md` for detailed documentation including:
- All available parameters
- GPU memory optimization tips
- Detailed troubleshooting
- Fine-tuning considerations

## 🎯 Typical Workflow

1. **Start with base model** to establish baseline performance:
   ```bash
   bash scripts/local_eval/run_qwen2vl_base_evaluation_a6000.sh
   ```

2. **Run fine-tuned model** to compare against baseline:
   ```bash
   bash scripts/local_eval/run_qwen2vl_evaluation_a6000.sh
   ```

3. **Compare results** using the evaluation metrics in the output JSON files.

## 💡 Tips

- Start with `--max-samples 10` to test everything works
- The base model evaluation is useful to understand what fine-tuning improves
- Both scripts automatically calculate evaluation metrics when `--enable-evaluation` is used
- Results include video filename, prompt, model output, and ground truth for easy comparison

