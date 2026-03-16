#!/bin/bash
#
# InternVL2.5-1B GRPO Training for Sign Language Translation
# Uses BLEU + BERTScore as rule-based rewards
# Starting from SFT checkpoint-2548

# Initialize conda
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate internvl_grpo
echo "Conda environment activated: internvl_grpo"

# Essential environment variables
export PYTHONPATH="/home/stu2/s15/mh2803/workspace/doc_sign_search/InternVL/internvl_chat:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1

# CUDA_HOME for DeepSpeed ops
export CUDA_HOME="$HOME/anaconda3/envs/internvl_grpo"

# Blackwell GPU: use gloo backend
export DIST_BACKEND=gloo

# NCCL settings (kept for reference)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Wandb
export WANDB_API_KEY="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"
export WANDB_PROJECT="internvl-grpo-sign-language"
export WANDB_RUN_NAME="grpo_1b_bleu1_bleu4_rouge_04_04_02_$(date +%m%d_%H%M)"

# Change to InternVL directory
cd /home/stu2/s15/mh2803/workspace/doc_sign_search/InternVL

# GPU configuration
GPU_IDS=${GPU_IDS:-"4,5,6,7"}
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# ── Model & Data ──
MODEL_PATH="/scratch/mh2803/checkpoints/finetune_internvl2_5_how2sign_1b_16fps_1218_121640/checkpoint-2548"
OUTPUT_DIR="/scratch/mh2803/checkpoints/grpo_internvl2_5_how2sign_1b_blackwell_bleu1_bleu4_rouge_04_04_02_0314"
DATA_PATH="/home/stu2/s15/mh2803/workspace/doc_sign_search/InternVL/data/how2sign/segmented_train_val_combined_sampled_10k.jsonl"
VIDEO_ROOT="/scratch/mh2803/train_crop_videos_224"

# ── GRPO parameters ──
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-128}
TEMPERATURE=${TEMPERATURE:-1.0}
BETA=${BETA:-0.0}

# ── Training parameters ──
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
NUM_EPOCHS=${NUM_EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:-0}  # 0 = use NUM_EPOCHS; set >0 for smoke test

# ── Video parameters ──
MAX_NUM_FRAME=${MAX_NUM_FRAME:-130}
MIN_NUM_FRAME=${MIN_NUM_FRAME:-32}
SAMPLING_METHOD='fps16.0'

# DeepSpeed Stage 0 for Blackwell
DEEPSPEED_CONFIG="internvl_chat/zero_stage0_config_blackwell.json"

# Launcher
export LAUNCHER=pytorch
export MASTER_PORT=29500

echo "GRPO Training for InternVL2.5-1B Sign Language Translation"
echo "=========================================================="
echo "SFT Checkpoint: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "GPU IDs: $GPU_IDS ($NUM_DEVICES GPUs)"
echo "Num Generations: $NUM_GENERATIONS"
echo "Max Completion Length: $MAX_COMPLETION_LENGTH"
echo "Temperature: $TEMPERATURE"
echo "Beta (KL): $BETA"
echo "Batch/Device: $BATCH_PER_DEVICE"
echo "Grad Accum Steps: $GRAD_ACCUM_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Frames: $MAX_NUM_FRAME"
echo "Sampling: $SAMPLING_METHOD"
echo ""

mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/grpo_training_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

deepspeed --include localhost:$GPU_IDS --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/grpo/internvl_grpo_train_bleu1_bleu4_rouge.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --video_root "$VIDEO_ROOT" \
    --conv_style internvl2_5 \
    --use_fast_tokenizer False \
    --do_train True \
    --num_train_epochs $NUM_EPOCHS \
    $([ "$MAX_STEPS" -gt 0 ] 2>/dev/null && echo "--max_steps $MAX_STEPS") \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --temperature $TEMPERATURE \
    --beta $BETA \
    --loss_type grpo \
    --scale_rewards group \
    --num_iterations 1 \
    --reward_weights_str "0.4,0.4,0.2" \
    --vision_select_layer -1 \
    --force_image_size 224 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --freeze_llm True \
    --freeze_backbone False \
    --freeze_mlp True \
    --use_llm_lora 16 \
    --bf16 True \
    --max_num_frame $MAX_NUM_FRAME \
    --min_num_frame $MIN_NUM_FRAME \
    --sampling_method "$SAMPLING_METHOD" \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 4 \
    --logging_steps 5 \
    --logging_first_step True \
    --report_to wandb \
    --run_name "$WANDB_RUN_NAME" \
    --grad_checkpoint True \
    --dataloader_num_workers 0 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --deepspeed "$DEEPSPEED_CONFIG" \
    2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "GRPO training completed successfully!"
    echo "Output: $OUTPUT_DIR"
else
    echo "GRPO training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check log: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi
