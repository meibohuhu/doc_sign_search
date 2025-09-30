#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=dpm_v2t
#SBATCH --error=/home/jw4905/powerful_vtr/RC_error/err_%j.txt
#SBATCH --output=/home/jw4905/powerful_vtr/RC_out/out_%j.txt
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-23:00:00
#SBATCH --partition tier3
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32g
#SBATCH --account=hsirecon
#SBATCH --partition=tier3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jw4905/conda/envs/jiamian_t2v/lib/
source ~/conda/etc/profile.d/conda.sh
conda activate jiamian_t2v
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

export DATASET_NAME="lambdalabs/pokemon-blip-captions"


python train_stochastic_nogen_unifymax.py --platform=RC  --use_gen=0 --raw_video   \
--diffusion_test_mode=benchmark    --kl_loss=kl_loss_nomean  \
--arch=clip_transformer_stochastic_nogen_bsxbstext_unifymax_1124version  --var_net=linear-cos  \
--exp_name=DiDeMo --videos_dir=/home/jw4905/text2video_data/Didemo/all_video/   \
--batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.4 --huggingface --dataset_name=DiDeMo  \
--evals_per_epoch=12  --prior_mean=0.0 --prior_var=0.01  --kl_loss_weight=0.0  \
 --mean_branch_weight=1.0  --std_weight=1.0  --stochasic_trials=20 --gpu=99 --num_epochs=10  \
 --gen_loss_genvid2txt_weight=0.8 --gen_loss_genvid2vid_weight 0.8 --stochastic_prior=uniform01 --uniform_scale=0.1   \
 --temperature=0.01 --trial_select=softmax  --temperature_test=0.01  --start_eval_epoch=5 --stochasic_trials_trn=10 --sims_out_type=enhanced_linear++



