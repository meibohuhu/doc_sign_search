#!/usr/bin/env python3
"""
Process how2sign validation videos to crop person and resize to 224x224.
Uses MediaPipe for fast person detection and cropping.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Paths
    input_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_val_segment_clips_24fps'
    output_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_val_segment_clips_stable_224x224'
    
    # Path to crop_video_sam.py script
    script_path = '/home/mh2803/projects/sign_language_llm/scripts/crop_video_sam.py'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found: {script_path}")
        sys.exit(1)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Build command - use bash to activate conda environment first
    cmd = f"""bash -c "source ~/.bashrc && conda activate qwenvl && python3 {script_path} --input-dir {input_dir} --output-dir {output_dir} --segmentation-backend mediapipe --resize-output 224 --crop-scale 1.2 --temporal-smoothing 0.7 --background-color black --mediapipe-model-selection 1 --workers 4" """
    
    print("🎬 Processing videos to crop person and resize to 224x224")
    print("=" * 60)
    print(f"📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 Backend: MediaPipe")
    print(f"📐 Size: 224x224")
    print("=" * 60)
    print()
    
    # Run the script
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n✅ Processing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Processing failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == '__main__':
    main()

