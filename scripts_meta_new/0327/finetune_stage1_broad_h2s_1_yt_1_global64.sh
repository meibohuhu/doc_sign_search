#!/bin/bash
#SBATCH --job-name=sft_1b_broad
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=100:00:00
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --error=logs/run_%j.err
#SBATCH --output=logs/run_%j.out

set -x

GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="/home/zachsun/doc_sign_search/InternVL/internvl_chat:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MASTER_PORT=34225
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_API_KEY="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"
export WANDB_PROJECT="internvl-sign-search"
export HF_HOME=/genai/fsx-project/zachsun

OUTPUT_DIR='/genai/fsx-project/zachsun/checkpoint/doc_sign_search/finetune_stage1_broad_h2s_1_yt_1_global64_0327'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Stage 1: Broad Mixed SFT ( H2S + YouTube)
# Architecture: InternViT-300M + MLP + InternLM2-1.8B
# Trainable: ViT (unfrozen) + LLM LoRA (rank 16), MLP frozen
# GPUs: 32
# Global Batch Size: 64 (per_device=2 x grad_accum=4 x 8 GPUs)
# Learning Rate: 5e-5
# Sampling: fps16.0
srun  \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  ${SRUN_ARGS} \
  python -u InternVL/internvl_chat/internvl/train/internvl_chat_finetune_local.py \
  --model_name_or_path "OpenGVLab/InternVL2_5-1B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/home/zachsun/doc_sign_search/script_meta_new/0327/train_stage1_meta_broad_h2s_1_yt_1.json" \
  --overwrite_output_dir True \
  --force_image_size 224 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --max_dynamic_patch 6 \
  --dynamic_image_size True \
  --min_num_frame 32 \
  --max_num_frame 160 \
  --sampling_method "fps16.0" \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone False \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 8 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 5 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --logging_first_step True \
  --max_seq_length 16584 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed /home/zachsun/doc_sign_search/InternVL/internvl_chat/zero_stage1_config.json \
  --report_to wandb \
  --run_name "finetune_stage1_broad_h2s_1_yt_1_global64_0327" \
  --use_packed_ds False \
  --use_data_resampling True \
  --dataloader_pin_memory True \
  --remove_unused_columns False \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
