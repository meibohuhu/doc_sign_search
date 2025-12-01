#!/usr/bin/env python3
"""
Extract vision encoder weights from MAE checkpoint and convert to standard InternVL format.
This script:
1. Loads the MAE checkpoint
2. Extracts only vision encoder weights (visual.* -> vision_model.*)
3. Saves as a new checkpoint compatible with standard InternVL training
"""

import torch
import os
import sys
import argparse
from pathlib import Path
from safetensors.torch import load_file, save_file

def extract_vision_encoder_from_mae(mae_checkpoint_path, output_path):
    """
    Extract vision encoder weights from MAE checkpoint
    
    Args:
        mae_checkpoint_path: Path to MAE checkpoint directory
        output_path: Path to save extracted vision encoder weights
    """
    print(f"📂 Loading MAE checkpoint from: {mae_checkpoint_path}")
    
    # Load model.safetensors
    model_path = os.path.join(mae_checkpoint_path, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"   Loading {model_path}...")
    state_dict = load_file(model_path)
    
    print(f"   ✅ Loaded {len(state_dict)} keys from checkpoint")
    
    # Extract vision encoder weights
    # MAE checkpoint uses "visual.*" prefix
    # Standard InternVL uses "vision_model.*" prefix
    vision_encoder_weights = {}
    
    print(f"\n🔄 Extracting vision encoder weights...")
    nan_inf_keys = []
    for key, value in state_dict.items():
        # Check if this is a vision encoder weight
        if key.startswith("visual."):
            # Convert "visual.*" to "vision_model.*"
            new_key = key.replace("visual.", "vision_model.", 1)
            # Check for NaN/Inf values
            if torch.is_tensor(value):
                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                if nan_count > 0:
                    nan_inf_keys.append((new_key, 'NaN', nan_count))
                elif inf_count > 0:
                    nan_inf_keys.append((new_key, 'Inf', inf_count))
            vision_encoder_weights[new_key] = value
        elif key.startswith("vision_model."):
            # Already in correct format
            # Check for NaN/Inf values
            if torch.is_tensor(value):
                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                if nan_count > 0:
                    nan_inf_keys.append((key, 'NaN', nan_count))
                elif inf_count > 0:
                    nan_inf_keys.append((key, 'Inf', inf_count))
            vision_encoder_weights[key] = value
    
    print(f"\n✅ Extracted {len(vision_encoder_weights)} vision encoder weights")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save vision encoder weights
    output_file = os.path.join(output_path, "vision_encoder_weights.safetensors")
    print(f"\n💾 Saving vision encoder weights to: {output_file}")
    save_file(vision_encoder_weights, output_file)
    
    # Also save as PyTorch format for compatibility
    output_pth = os.path.join(output_path, "vision_encoder_weights.pth")
    print(f"💾 Also saving as PyTorch format: {output_pth}")
    torch.save(vision_encoder_weights, output_pth)
    
    # Create a minimal checkpoint structure for loading
    print(f"\n📋 Creating checkpoint structure...")
    
    # Copy trainer_state.json if exists (for resume training info)
    trainer_state_src = os.path.join(mae_checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_src):
        import shutil
        trainer_state_dst = os.path.join(output_path, "trainer_state.json")
        shutil.copy2(trainer_state_src, trainer_state_dst)
        print(f"   ✅ Copied trainer_state.json")
    
    print(f"\n✅ Vision encoder weights extracted successfully!")
    print(f"📁 Output directory: {output_path}")
    print(f"📄 Files created:")
    print(f"   - {output_file}")
    print(f"   - {output_pth}")
    
    # Show summary
    print(f"\n📊 Summary:")
    print(f"   Total keys in MAE checkpoint: {len(state_dict)}")
    print(f"   Vision encoder keys extracted: {len(vision_encoder_weights)}")
    print(f"   Keys with 'visual.' prefix: {sum(1 for k in state_dict.keys() if k.startswith('visual.'))}")
    print(f"   Keys with 'vision_model.' prefix: {sum(1 for k in state_dict.keys() if k.startswith('vision_model.'))}")
    if nan_inf_keys:
        print(f"\n⚠️  WARNING: Found {len(nan_inf_keys)} keys with NaN/Inf values:")
        for key, issue_type, count in nan_inf_keys:
            print(f"   {key}: {issue_type} ({count} values)")
        print(f"   These keys will be skipped during model loading.")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract vision encoder from MAE checkpoint")
    parser.add_argument(
        "--mae-checkpoint",
        type=str,
        required=True,
        help="Path to MAE checkpoint directory (e.g., checkpoint-30000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save extracted vision encoder weights"
    )
    
    args = parser.parse_args()
    
    extract_vision_encoder_from_mae(args.mae_checkpoint, args.output)


if __name__ == "__main__":
    main()

