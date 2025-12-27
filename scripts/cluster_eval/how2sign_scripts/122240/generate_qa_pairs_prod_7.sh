#!/bin/bash
#
# Generate QA Pairs from GPT-4V/GPT-5 Evaluation Results
# Uses Azure OpenAI GPT-5/GPT-4o to generate QA pairs from statements
#

# Set up conda environment (if needed)
# source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda activate your_env_name
# echo "✅ Conda environment activated"

# Essential environment variables
export PYTHONUNBUFFERED=1

# Change to project directory
cd /code/doc_sign_search

# Configuration
MODEL="${MODEL:-gpt-5}"  # Azure deployment model name
INPUT_FILE="${INPUT_FILE:-/code/doc_sign_search/outputs/gpt5_eval/122240/122240_7/gpt4v_results_20251226_105115.json}"
OUTPUT_FILE="${OUTPUT_FILE:-}"  # Auto-generated if not specified
OUT_DIR="${OUT_DIR:-/code/doc_sign_search/outputs/gpt5_eval/122240/122240_7/}"

# Azure OpenAI Configuration
AZURE_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://dil-research-3.openai.azure.com/}"
AZURE_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
AZURE_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-$MODEL}"  # Default to MODEL if not set

# Generation parameters
MAX_SAMPLES=${MAX_SAMPLES:-}  # Set to a number to limit samples, empty for full processing
MAX_TOKENS=${MAX_TOKENS:-16384}  # Max tokens to generate
TEMPERATURE=${TEMPERATURE:-1}  # Sampling temperature
MAX_WORKERS=${MAX_WORKERS:-2}  # Maximum number of worker threads
SAVE_INTERVAL=${SAVE_INTERVAL:-10}  # Save results every N samples
NO_EVALUATE=${NO_EVALUATE:-false}  # Set to "true" to disable QA pair evaluation

echo "🎬 Generate QA Pairs from GPT-4V/GPT-5 Results"
echo "=============================================="
echo "Azure Endpoint: $AZURE_ENDPOINT"
echo "Azure API Version: $AZURE_API_VERSION"
echo "Azure Deployment: $AZURE_DEPLOYMENT"
echo "Model: $MODEL"
echo "Input File: $INPUT_FILE"
echo "Output Dir: $OUT_DIR"
if [ -n "$OUTPUT_FILE" ]; then
    echo "Output File: $OUTPUT_FILE"
else
    echo "Output File: (auto-generated)"
fi
echo "Max Tokens: $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Max Workers: $MAX_WORKERS"
echo "Save Interval: $SAVE_INTERVAL"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples (limited): $MAX_SAMPLES"
else
    echo "Max Samples: Full processing"
fi
if [ "$NO_EVALUATE" = "true" ]; then
    echo "QA Evaluation: DISABLED"
else
    echo "QA Evaluation: ENABLED (default)"
fi
echo ""

# Check if Azure API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY (Azure subscription key) environment variable is not set"
    echo "   You can set it with: export OPENAI_API_KEY='your-azure-subscription-key'"
    echo "   Or pass it via --api-key argument"
    echo ""
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"
echo "📁 Output directory: $OUT_DIR"
echo ""

# Log file with timestamp
LOG_FILE="${OUT_DIR}/qa_generation_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"
echo ""

# Build command arguments
GEN_ARGS=(
    --model "$MODEL"
    --input-file "$INPUT_FILE"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --max-workers "$MAX_WORKERS"
    --save-interval "$SAVE_INTERVAL"
    --azure-endpoint "$AZURE_ENDPOINT"
    --azure-api-version "$AZURE_API_VERSION"
    --azure-deployment "$AZURE_DEPLOYMENT"
)

# Add API key if provided as environment variable
if [ -n "$OPENAI_API_KEY" ]; then
    GEN_ARGS+=(--api-key "$OPENAI_API_KEY")
fi

# Add output file if specified
if [ -n "$OUTPUT_FILE" ]; then
    GEN_ARGS+=(--output-file "$OUTPUT_FILE")
fi

# Add max-samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    GEN_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# Add no-evaluate if enabled
if [ "$NO_EVALUATE" = "true" ]; then
    GEN_ARGS+=(--no-evaluate)
fi

# Get python path (use system python or conda python)
PYTHON_CMD="python3"
if command -v python3 &> /dev/null; then
    PYTHON_CMD=$(which python3)
elif command -v python &> /dev/null; then
    PYTHON_CMD=$(which python)
else
    echo "❌ Error: Python not found!"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Run QA pair generation
echo "🚀 Starting QA pair generation..."
echo ""

$PYTHON_CMD scripts/cluster_eval/how2sign_scripts/generate_qa_pairs_from_statements.py \
    "${GEN_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

GEN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $GEN_EXIT_CODE -eq 0 ]; then
    echo "✅ QA pair generation completed successfully!"
    echo "📁 Results saved to: $OUT_DIR"
    echo "📝 Log file: $LOG_FILE"
else
    echo "❌ QA pair generation failed with exit code $GEN_EXIT_CODE."
    echo "Please check the error messages above for details."
    echo "Check log file: $LOG_FILE"
    exit $GEN_EXIT_CODE
fi

