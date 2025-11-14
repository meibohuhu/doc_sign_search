import cv2
import os
import glob
import subprocess
import shutil
from pathlib import Path

# Configuration
input_folder = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips'
output_folder = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_24fps'
target_fps = 24
fps_tolerance = 0.1  # Consider videos within ±0.1 fps as already correct

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Try to find ffmpeg in various locations
def find_ffmpeg():
    """Find ffmpeg binary in common locations"""
    # Check conda environment first
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        conda_ffmpeg = os.path.join(conda_prefix, 'bin', 'ffmpeg')
        if os.path.exists(conda_ffmpeg):
            return conda_ffmpeg
    
    # Check system PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Check common locations
    common_paths = [
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/opt/ffmpeg/bin/ffmpeg',
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    raise RuntimeError('Could not find ffmpeg. Please install it or ensure it is in PATH.')

# Get ffmpeg path
try:
    ffmpeg_cmd = find_ffmpeg()
    print(f'Using ffmpeg: {ffmpeg_cmd}')
except RuntimeError as e:
    print(f'Error: {e}')
    print('Attempting to use system ffmpeg anyway...')
    ffmpeg_cmd = 'ffmpeg'

# Check if libiconv is available
def check_libiconv():
    """Check if libiconv is available and return library path if found"""
    import ctypes.util
    
    # Try to find libiconv
    libiconv_path = ctypes.util.find_library('iconv')
    if libiconv_path:
        return os.path.dirname(libiconv_path)
    
    # Check common locations
    common_paths = [
        '/usr/lib',
        '/usr/lib64',
        '/usr/local/lib',
        '/usr/local/lib64',
    ]
    
    for path in common_paths:
        iconv_files = glob.glob(os.path.join(path, '*iconv*'))
        if iconv_files:
            return path
    
    return None

libiconv_path = check_libiconv()
if not libiconv_path:
    print('⚠️  Warning: libiconv library not found.')
    print('   If you encounter errors, try: conda install -c conda-forge libiconv')
else:
    print(f'✓ Found libiconv in: {libiconv_path}')

# Get all video files
videos = glob.glob(os.path.join(input_folder, '*.mp4'))
print(f'Found {len(videos)} videos')
print(f'Target FPS: {target_fps}')
print(f'Output folder: {output_folder}\n')

# Statistics
already_24fps = []
needs_conversion = []
converted = []
failed = []

# Process each video
for i, video_path in enumerate(videos, 1):
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_folder, video_name)
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f'[{i}/{len(videos)}] Skipping {video_name} (already exists)')
        continue
    
    # Get current FPS
    cap = cv2.VideoCapture(video_path)
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Check if conversion is needed
    if abs(current_fps - target_fps) < fps_tolerance:
        # Video is already at target FPS, just copy it
        try:
            import shutil
            shutil.copy2(video_path, output_path)
            already_24fps.append(video_name)
            print(f'[{i}/{len(videos)}] Copied {video_name} (already {current_fps:.2f} fps)')
        except Exception as e:
            failed.append((video_name, str(e)))
            print(f'[{i}/{len(videos)}] ERROR copying {video_name}: {e}')
    else:
        # Need to convert
        needs_conversion.append((video_name, current_fps))
        print(f'[{i}/{len(videos)}] Converting {video_name} ({current_fps:.2f} fps -> {target_fps} fps)...', end=' ', flush=True)
        
        try:
            # Use ffmpeg to convert FPS
            # -i: input file
            # -filter:v "fps=fps=24": set frame rate filter
            # -c:v libx264: video codec
            # -c:a copy: copy audio without re-encoding (faster, preserves quality)
            # -y: overwrite output file if exists
            cmd = [
                ffmpeg_cmd,
                '-i', video_path,
                '-filter:v', f'fps=fps={target_fps}',
                '-c:v', 'libx264',
                '-preset', 'medium',  # Balance between speed and compression
                '-crf', '23',  # Quality setting (18-28 is good, 23 is default)
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y',  # Overwrite output
                output_path
            ]
            
            # Set up environment with library paths
            env = os.environ.copy()
            
            # Try to find libiconv library paths
            lib_paths = []
            
            # Add libiconv path if found earlier
            if libiconv_path:
                lib_paths.append(libiconv_path)
            
            # Check conda environment
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            if conda_prefix:
                conda_lib = os.path.join(conda_prefix, 'lib')
                if os.path.exists(conda_lib) and conda_lib not in lib_paths:
                    lib_paths.append(conda_lib)
            
            # Check spack environments (if using spack)
            spack_root = os.environ.get('SPACK_ROOT', '')
            if spack_root:
                # Look for libiconv in spack
                spack_lib_patterns = [
                    f'{spack_root}/opt/spack/*/*/libiconv/*/lib',
                    f'{spack_root}/opt/spack/*/*/gettext/*/lib',  # gettext often includes libiconv
                ]
                for pattern in spack_lib_patterns:
                    found_paths = glob.glob(pattern)
                    for path in found_paths:
                        if path not in lib_paths:
                            lib_paths.append(path)
            
            # Add common system paths
            common_lib_paths = [
                '/usr/lib',
                '/usr/lib64',
                '/usr/local/lib',
                '/usr/local/lib64',
                '/lib',
                '/lib64',
            ]
            for path in common_lib_paths:
                if os.path.exists(path) and path not in lib_paths:
                    lib_paths.append(path)
            
            # Update LD_LIBRARY_PATH
            current_ld_path = env.get('LD_LIBRARY_PATH', '')
            if lib_paths:
                new_ld_path = ':'.join(lib_paths)
                if current_ld_path:
                    env['LD_LIBRARY_PATH'] = f'{new_ld_path}:{current_ld_path}'
                else:
                    env['LD_LIBRARY_PATH'] = new_ld_path
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                env=env
            )
            
            converted.append((video_name, current_fps))
            print('✓')
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            failed.append((video_name, error_msg))
            print('✗ FAILED')
            
            # Check for libiconv error specifically
            if 'libiconv' in error_msg.lower():
                print(f'  Error: Missing libiconv library')
                print(f'  Try: conda install -c conda-forge ffmpeg')
                print(f'  Or: Set LD_LIBRARY_PATH to include libiconv library location')
            else:
                # Print first few lines of error
                error_lines = error_msg.split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        print(f'  Error: {line}')
        except Exception as e:
            failed.append((video_name, str(e)))
            print(f'✗ FAILED: {e}')

# Print summary
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'Total videos processed: {len(videos)}')
print(f'Already {target_fps} fps (copied): {len(already_24fps)}')
print(f'Converted to {target_fps} fps: {len(converted)}')
print(f'Failed: {len(failed)}')

if failed:
    print('\nFailed videos:')
    for video_name, error in failed:
        print(f'  - {video_name}: {error[:100]}...')

if needs_conversion:
    print(f'\nVideos that were converted:')
    for video_name, old_fps in needs_conversion[:10]:  # Show first 10
        print(f'  - {video_name}: {old_fps:.2f} fps -> {target_fps} fps')
    if len(needs_conversion) > 10:
        print(f'  ... and {len(needs_conversion) - 10} more')

print(f'\nOutput folder: {output_folder}')
print('=' * 70)
