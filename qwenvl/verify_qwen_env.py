#!/usr/bin/env python
"""
Verification script for Qwen2.5-VL training environment
Run this after setting up the environment to ensure everything works correctly.
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name} import failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Qwen2.5-VL Environment Verification")
    print("=" * 60)
    
    all_passed = True
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info[:2] != (3, 10):
        print("⚠ Warning: Expected Python 3.10")
        all_passed = False
    else:
        print("✓ Python version correct")
    
    # Check PyTorch
    print("\n--- PyTorch ---")
    if check_import("torch"):
        import torch
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ⚠ Warning: CUDA not available")
            all_passed = False
    else:
        all_passed = False
    
    # Check Flash Attention
    print("\n--- Flash Attention ---")
    if check_import("flash_attn", "flash-attn"):
        import flash_attn
        print(f"  Version: {flash_attn.__version__}")
    else:
        all_passed = False
    
    # Check Transformers
    print("\n--- Transformers ---")
    if check_import("transformers"):
        import transformers
        print(f"  Version: {transformers.__version__}")
        if transformers.__version__ != "4.51.3":
            print(f"  ⚠ Warning: Expected version 4.51.3, got {transformers.__version__}")
    else:
        all_passed = False
    
    # Check Accelerate
    print("\n--- Accelerate ---")
    if check_import("accelerate"):
        import accelerate
        print(f"  Version: {accelerate.__version__}")
    else:
        all_passed = False
    
    # Check DeepSpeed
    print("\n--- DeepSpeed ---")
    if check_import("deepspeed"):
        import deepspeed
        print(f"  Version: {deepspeed.__version__}")
    else:
        all_passed = False
    
    # Check PEFT
    print("\n--- PEFT ---")
    if check_import("peft"):
        import peft
        print(f"  Version: {peft.__version__}")
    else:
        all_passed = False
    
    # Check other packages
    print("\n--- Additional Packages ---")
    check_import("ujson") and all_passed
    check_import("liger_kernel") and all_passed
    check_import("datasets") and all_passed
    check_import("wandb") and all_passed
    check_import("decord") and all_passed
    
    # Check Qwen VL Utils
    print("\n--- Qwen VL Utils ---")
    try:
        from qwen_vl_utils import process_vision_info
        print("✓ qwen-vl-utils imported successfully")
    except ImportError as e:
        print(f"✗ qwen-vl-utils import failed: {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! Environment is ready for training.")
    else:
        print("✗ Some checks failed. Please review the issues above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
