#!/bin/bash
#
# Helper script to run Qwen2VL evaluation with nohup
# Usage: ./run_evaluation_nohup.sh [GPU_ID] [MAX_SAMPLES]
# Example: ./run_evaluation_nohup.sh 0
#

# Get arguments
GPU_ID=${1:-"1"}  # Default to GPU 1
MAX_SAMPLES=${2:-""}  # Default to empty (use script default)

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/local1/mhu/sign_language_llm"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_qwen2vl_how2sign_2xa6000.sh"

# Generate nohup output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NOHUP_LOG="${PROJECT_ROOT}/nohup_qwen2vl_eval_${TIMESTAMP}.log"

# Change to project root
cd "$PROJECT_ROOT"

echo "🚀 Starting Qwen2VL evaluation with nohup..."
echo "📌 GPU ID: $GPU_ID"
echo "📌 Max Samples: ${MAX_SAMPLES:-"Default"}"
echo "📝 Nohup log: $NOHUP_LOG"
echo ""

# Build command with environment variables
ENV_VARS="GPU_IDS=\"${GPU_ID}\""
if [ -n "$MAX_SAMPLES" ]; then
    ENV_VARS="${ENV_VARS} MAX_SAMPLES=${MAX_SAMPLES}"
fi

# Run with nohup
nohup bash -c "${ENV_VARS} bash ${EVAL_SCRIPT}" > "$NOHUP_LOG" 2>&1 &

# Get the process ID
PID=$!

echo "✅ Evaluation started in background!"
echo "   Process ID: $PID"
echo "   Nohup log: $NOHUP_LOG"
echo ""
echo "📋 Useful commands:"
echo "   View nohup output:    tail -f $NOHUP_LOG"
echo "   Check process:        ps aux | grep $PID"
echo "   Stop evaluation:      kill $PID"
echo ""
echo "💡 Note: The script also creates a detailed log in outputs/qwen2vl_eval/"
echo "   View it with: tail -f outputs/qwen2vl_eval/evaluation_*.log"

