#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=llavaov_ssvp_train
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1-12:00:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --partition tier3
#SBATCH --mem=256g


spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

conda activate mh_llava

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO


cd /home/mh2803/projects/sign_language_llm/llava

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=4 --nnodes=1 --master_port=29501 train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 1e-5 \
    --deepspeed zero3.json \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --version qwen_1_5 \
    --data_path /home/mh2803/projects/sign_language_llm/vanshika/asl_test/train_ssvp.json \
    --video_folder /home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/videos/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter" \
    --mm_vision_tower_lr=1e-5 \
    --vision_tower mobileclip_l_1024 \
    --mm_projector_path /home/mh2803/projects/sign_language_llm/llava/checkpoints/pretrain_fastvit_llavaov/checkpoint-16000/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 3072 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_2 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name llavaov_ssvp_train \
    --output_dir /home/mh2803/projects/sign_language_llm/llava/checkpoints/llavaov_ssvp_finetune/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 5 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    --dataloader_drop_last True \
    --frames_upbound 8 \
    --mm_newline_position grid \
