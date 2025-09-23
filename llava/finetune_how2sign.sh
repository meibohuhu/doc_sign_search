#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=how2sign_fastvit_train
#SBATCH --error=/home/vp1837/data/LLaVA-NeXT/RC_error/err_%j.txt
#SBATCH --output=/home/vp1837/data/LLaVA-NeXT/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-05:30:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --partition tier3
#SBATCH --mem=128g
#SBATCH --account=ai-asl


spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

conda activate llavaov

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

ACCELERATE_DISABLE_CPU_AFFINITY=1 torchrun --nproc_per_node=4 --nnodes=1 --master_port=29501 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 1e-6 \
    --model_name_or_path /home/vp1837/data/LLaVA-NeXT/checkpoints/merged_ssvp_fastvit/ \
    --version qwen_1_5 \
    --data_path /home/vp1837/data/asl/train/rgb_videos_df_ssh.json \
    --video_folder /home/vp1837/train/raw_videos/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower mobileclip_l_1024 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_1 \
    --image_grid_pinpoints "(1x1),(2x2),(3x3),(4x4),(5x5),(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name how2sign_fastvit_train \
    --output_dir /home/vp1837/data/LLaVA-NeXT/checkpoints/how2sign_fastvit_llavaov/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 5 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --frames_upbound 3 




    
   
    