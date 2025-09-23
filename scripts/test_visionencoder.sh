#!/bin/bash -l
#SBATCH --job-name=dryrun_ssvp
#SBATCH --error=/home/vp1837/data/LLaVA-NeXT/RC_error/dryrun_err_%j.txt
#SBATCH --output=/home/vp1837/data/LLaVA-NeXT/RC_out/dryrun_out_%j.txt
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:05:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=debug
#SBATCH --mem=32g
#SBATCH --account=ai-asl

# Load CUDA
spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Activate Conda environment
conda activate llavaov

# Python dry-run check
python << 'PYCODE'
from llava.model.builder import load_pretrained_model

model_path = "/home/vp1837/data/LLaVA-NeXT/checkpoints/merged_ssvp_fastvit/"
model_name = "llava-qwen-lora-0.5b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device_map="auto",
    torch_dtype="float16",
    attn_implementation="eager",
)

vision_tower = model.get_vision_tower()
print("\n=== DRY RUN CHECK ===")
print("Vision tower object:", vision_tower)
print("Vision tower config:", getattr(model.config, "mm_vision_tower", None))
print("Vision tower image size:", getattr(vision_tower.config, "vision_tower_image_size", None))
print("Image processor:", image_processor)
print("=====================\n")
PYCODE
