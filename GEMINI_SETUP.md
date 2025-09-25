# Gemini 2.5 Pro Setup Instructions

## Installation

1. Install the required package in the qwen25_vl_sign environment:
```bash
/home/ztao/anaconda3/envs/qwen25_vl_sign/bin/pip install google-generativeai>=0.8.0
```

Or install from requirements:
```bash
/home/ztao/anaconda3/envs/qwen25_vl_sign/bin/pip install -r requirements_gemini.txt
```

**Note**: The package is already installed in your environment! ✅

## API Key Setup

1. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

2. Set the environment variable:
```bash
export GEMINI_API_KEY='your_api_key_here'
```

## Usage

### Run with all samples:
```bash
bash scripts/eval/run_gemini25_test.sh
```

### Run with limited samples:
```bash
bash scripts/eval/run_gemini25_test.sh 5
```

### Run directly with Python:
```bash
python -m playground.demo.gemini25_metrics \
    --api-key "your_api_key_here" \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos_test.json \
    --video-folder "/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/" \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --max-samples 1
```

## Output

Results will be saved to:
- `/local1/mhu/LLaVANeXT_RC/new_outputs/gemini25_results.json` - Model outputs
- `/local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics.json` - Evaluation metrics

## Notes

- The script uses Gemini 2.5 Pro (latest model) with fallback to Gemini 1.5 Pro
- Video processing extracts 4-8 frames depending on video size
- API rate limits may apply based on your Google AI Studio quota
- Gemini 2.5 Pro provides improved performance and capabilities
