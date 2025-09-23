import os

# Manual imports to ensure eval() doesn't fail
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
from llava.model.language_model.llava_mixtral import LlavaMixtralForCausalLM, LlavaMixtralConfig

AVAILABLE_MODELS = {
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
}

# You can keep this block if you want to still print warnings for future models
for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")


# Update global namespace (this requires actual imports to work)
globals().update({cls_name.strip(): eval(cls_name.strip()) for model_classes in AVAILABLE_MODELS.values() for cls_name in model_classes.split(",")})
