#!/usr/bin/env python3
"""
Load vision encoder weights into a standard InternVL checkpoint structure.
This creates a checkpoint that can be used with --resume_from_checkpoint,
but only initializes the vision encoder from MAE weights.

Usage:
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate internvl
    python3 load_vision_encoder_to_checkpoint.py --vision-encoder-weights <path> --output <path>
"""

import torch
import os
import sys
import json
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

# Add InternVL to path
internvl_path = os.path.join(os.path.dirname(__file__), '../../internvl_chat')
if os.path.exists(internvl_path):
    sys.path.insert(0, internvl_path)

def create_checkpoint_with_vision_encoder(
    vision_encoder_weights_path,
    base_model_name,
    output_checkpoint_path,
    trainer_state_path=None
):
    """
    Create a checkpoint with vision encoder weights loaded from MAE checkpoint.
    
    Args:
        vision_encoder_weights_path: Path to extracted vision encoder weights (.safetensors or .pth)
        base_model_name: Base model name (e.g., "OpenGVLab/InternVL2_5-2B")
        output_checkpoint_path: Where to save the new checkpoint
        trainer_state_path: Optional path to trainer_state.json (for training info)
    """
    print(f"📂 Loading vision encoder weights from: {vision_encoder_weights_path}")
    
    # Load vision encoder weights
    if vision_encoder_weights_path.endswith('.safetensors'):
        vision_weights = load_file(vision_encoder_weights_path)
    else:
        vision_weights = torch.load(vision_encoder_weights_path, map_location='cpu')
    
    print(f"   ✅ Loaded {len(vision_weights)} vision encoder weights")
    
    # Load base model to get full state dict structure
    print(f"\n📦 Loading base model: {base_model_name}")
    try:
        from internvl.model.internvl_chat import InternVLChatModel
        model = InternVLChatModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print(f"   ✅ Base model loaded")
    except Exception as e:
        print(f"   ❌ Error loading base model: {e}")
        raise
    
    # Get full state dict
    print(f"\n🔄 Merging vision encoder weights into model...")
    full_state_dict = model.state_dict()
    
    # Replace vision_model.* weights with extracted weights
    replaced_count = 0
    for key in list(full_state_dict.keys()):
        if key.startswith('vision_model.'):
            # Check if we have this weight in vision_weights
            if key in vision_weights:
                full_state_dict[key] = vision_weights[key]
                replaced_count += 1
            else:
                print(f"   ⚠️  Warning: {key} not found in vision encoder weights, keeping base model weight")
    
    print(f"   ✅ Replaced {replaced_count} vision encoder weights")
    
    # Create output directory
    os.makedirs(output_checkpoint_path, exist_ok=True)
    
    # Save model weights
    model_file = os.path.join(output_checkpoint_path, "model.safetensors")
    print(f"\n💾 Saving model to: {model_file}")
    save_file(full_state_dict, model_file)
    
    # Also save as PyTorch format for compatibility
    model_pth = os.path.join(output_checkpoint_path, "pytorch_model.bin")
    print(f"💾 Also saving as PyTorch format: {model_pth}")
    torch.save(full_state_dict, model_pth)
    
    # Copy trainer_state.json if provided
    if trainer_state_path and os.path.exists(trainer_state_path):
        trainer_state_dst = os.path.join(output_checkpoint_path, "trainer_state.json")
        shutil.copy2(trainer_state_path, trainer_state_dst)
        print(f"   ✅ Copied trainer_state.json")
        
        # Update trainer_state to indicate this is a fresh start with custom vision encoder
        with open(trainer_state_dst, 'r') as f:
            trainer_state = json.load(f)
        trainer_state['log_history'] = []  # Clear training history
        trainer_state['global_step'] = 0
        trainer_state['epoch'] = 0.0
        with open(trainer_state_dst, 'w') as f:
            json.dump(trainer_state, f, indent=2)
        print(f"   ✅ Reset training state (global_step=0, epoch=0)")
    
    print(f"\n✅ Checkpoint created successfully!")
    print(f"📁 Output directory: {output_checkpoint_path}")
    print(f"\n💡 Usage:")
    print(f"   RESUME_FROM_CHECKPOINT=\"{output_checkpoint_path}\" bash your_training_script.sh")
    print(f"\n⚠️  Note: This checkpoint initializes vision encoder from MAE weights,")
    print(f"   but training will start from step 0 (not resume from step 30000)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create checkpoint with MAE vision encoder weights")
    parser.add_argument(
        "--vision-encoder-weights",
        type=str,
        required=True,
        help="Path to extracted vision encoder weights (.safetensors or .pth)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="OpenGVLab/InternVL2_5-2B",
        help="Base model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output checkpoint directory"
    )
    parser.add_argument(
        "--trainer-state",
        type=str,
        default=None,
        help="Optional: Path to trainer_state.json from original checkpoint"
    )
    
    args = parser.parse_args()
    
    create_checkpoint_with_vision_encoder(
        args.vision_encoder_weights,
        args.base_model,
        args.output,
        args.trainer_state
    )


if __name__ == "__main__":
    main()

