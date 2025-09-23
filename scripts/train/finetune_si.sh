#!/bin/bash -l
# NOTE the -l flag!

#SBATCH --job-name=llava-video-rgb
#SBATCH --error=/home/vp1837/data/llava_video/RC_error/err_%j.txt
#SBATCH --output=/home/vp1837/data/llava_video/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-5:30:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --partition tier3
#SBATCH --mem=128g
#SBATCH --account=ai-asl

spack load cuda@12.4.0/obxqih4
conda activate llava

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=4 --nnodes=1 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-7b-si \
    --version qwen_1_5 \
    --data_path /home/vp1837/data/asl/train/rgb_videos_df_ov.json \
    --video_folder /home/vp1837/train/raw_videos/ \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name llava-video-rgb-1epo \
    --output_dir /home/vp1837/data/LLaVA-NeXT/checkpoints/llvideo-rgb-1epo/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 10
