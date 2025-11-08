#!/usr/bin/env python3
"""
Check if model has necessary modules for importance calculation
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ["DISABLE_FLASH_ATTN"] = "1"

qwen_finetune_path = '/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src'
if os.path.exists(qwen_finetune_path):
    sys.path.insert(0, qwen_finetune_path)

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

checkpoint_path = '/local1/mhu/sign_language_llm/outputs/checkpoints'

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

non_lora_path = os.path.join(checkpoint_path, 'non_lora_state_dict.bin')
if os.path.exists(non_lora_path):
    state_dict = torch.load(non_lora_path, map_location='cpu', weights_only=False)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith('module.'):
            new_key = new_key[7:]
        if new_key.startswith('base_model.model.model.'):
            new_key = new_key[23:]
        elif new_key.startswith('base_model.model.'):
            new_key = new_key[17:]
        elif new_key.startswith('model.'):
            new_key = new_key[6:]
        cleaned_state_dict[new_key] = v
    model.load_state_dict(cleaned_state_dict, strict=False)

adapter_path = os.path.join(checkpoint_path, 'adapter_config.json')
if os.path.exists(adapter_path):
    model = PeftModel.from_pretrained(model, checkpoint_path)

print("\nChecking for importance-related modules...")

# Get actual model (handle PeftModel wrapper)
actual_model = model
if hasattr(model, 'get_base_model'):
    actual_model = model.get_base_model()
elif hasattr(model, 'base_model'):
    actual_model = model.base_model

# Check for mean_logvar_lgkld
has_mean_logvar = False
if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'mean_logvar_lgkld'):
    has_mean_logvar = True
    print("   ✅ Found mean_logvar_lgkld at model.model.mean_logvar_lgkld")
elif hasattr(actual_model, 'mean_logvar_lgkld'):
    has_mean_logvar = True
    print("   ✅ Found mean_logvar_lgkld at model.mean_logvar_lgkld")

# Check for prior_mu and prior_logvar
has_prior = False
if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'prior_mu') and hasattr(actual_model.model, 'prior_logvar'):
    has_prior = True
    print("   ✅ Found prior_mu and prior_logvar at model.model")
elif hasattr(actual_model, 'prior_mu') and hasattr(actual_model, 'prior_logvar'):
    has_prior = True
    print("   ✅ Found prior_mu and prior_logvar at model")

if not has_mean_logvar:
    print("   ❌ mean_logvar_lgkld not found")
if not has_prior:
    print("   ❌ prior_mu and prior_logvar not found")

print("\nModel structure:")
print(f"   Model type: {type(actual_model)}")
if hasattr(actual_model, 'model'):
    print(f"   Has model.model: {type(actual_model.model)}")
    if hasattr(actual_model.model, '__dict__'):
        attrs = [k for k in dir(actual_model.model) if not k.startswith('_') and 'mean' in k.lower() or 'prior' in k.lower() or 'logvar' in k.lower()]
        if attrs:
            print(f"   Related attributes: {attrs[:10]}")

print("\nChecking checkpoint keys...")
if os.path.exists(non_lora_path):
    state_dict = torch.load(non_lora_path, map_location='cpu', weights_only=False)
    importance_keys = [k for k in state_dict.keys() if 'mean' in k.lower() or 'prior' in k.lower() or 'logvar' in k.lower() or 'lkld' in k.lower()]
    if importance_keys:
        print(f"   Found {len(importance_keys)} importance-related keys:")
        for k in importance_keys[:10]:
            print(f"      {k}")
    else:
        print("   ❌ No importance-related keys found in checkpoint")



