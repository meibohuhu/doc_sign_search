# Installation Guide for GPT-5 Vision Evaluation Script

This guide explains how to set up the environment to run `gpt4v_evaluation_how2sign_prod.py` on a new cluster.

## Required Python Version

- Python 3.8 or higher (Python 3.10+ recommended)

## Installation Steps

### 1. Create a virtual environment (recommended)

```bash
# Using venv
python3 -m venv venv_gpt4v_eval
source venv_gpt4v_eval/bin/activate  # On Linux/Mac
# or
venv_gpt4v_eval\Scripts\activate  # On Windows

# Or using conda
conda create -n gpt4v_eval python=3.10
conda activate gpt4v_eval
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements_gpt4v_eval.txt
```

### 3. Verify installation

```bash
python3 -c "import cv2; import numpy; import PIL; import tqdm; from openai import AzureOpenAI; print('✅ All dependencies installed successfully!')"
```

## Required Libraries

The script requires the following Python packages:

1. **opencv-python** (cv2) - For video frame extraction
2. **numpy** - For numerical operations
3. **Pillow** (PIL) - For image processing
4. **tqdm** - For progress bars
5. **openai** - For Azure OpenAI API access

## Environment Variables

Set the following environment variables before running:

```bash
export OPENAI_API_KEY="your-azure-subscription-key"
export AZURE_OPENAI_ENDPOINT="https://dil-research-3.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-5"  # Or your deployment name
```

## Quick Test

Test the installation by running:

```bash
python3 scripts/cluster_eval/how2sign_scripts/gpt4v_evaluation_how2sign_prod.py --help
```

## Troubleshooting

### OpenCV installation issues

If `opencv-python` fails to install, try:
```bash
pip install opencv-python-headless  # Lighter version without GUI support
```

### Azure OpenAI connection issues

Make sure:
1. Your Azure subscription key is valid
2. The endpoint URL is correct
3. The deployment name matches your Azure resource
4. Your IP is whitelisted (if required by your Azure setup)



