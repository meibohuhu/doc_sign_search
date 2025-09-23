import torch
import torch.nn as nn
from transformers import AutoTokenizer
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

# === Load base model ===
model = LlavaQwenForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16
)

# === Manually create mm_projector ===
# Pretraining used mlp2x_gelu: input=1024 (FastViT), output=4096 (Qwen2)
model.model.mm_projector = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096)
)

# === Load projector weights ===
projector_path = "/home/vp1837/data/LLaVA-NeXT/checkpoints/pretrain_fastvit_llavaov/mm_projector.bin"
raw_state_dict = torch.load(projector_path, map_location="cpu")
stripped_state_dict = {k.replace("model.mm_projector.", ""): v for k, v in raw_state_dict.items()}
model.model.mm_projector.load_state_dict(stripped_state_dict)


# === Save final model ===
output_path = "/home/vp1837/data/LLaVA-NeXT/checkpoints/merged_fullmodel"
model.save_pretrained(output_path)

# === Save tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print("Model merged and saved with mm_projector.")
