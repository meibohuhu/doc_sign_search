"""
Download videos from youtube_video_ids_stage2.txt
"""
import os
import subprocess
import sys
import time
from pathlib import Path

def read_video_ids(file_path):
    """Read video IDs from file"""
    video_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            video_id = line.strip()
            if video_id:
                video_ids.append(video_id)
    return video_ids

def is_valid_video_file(file_path, require_mp4=False):
    """Check if a file is a valid video file (MP4, TS, MKV, WebM)
    
    Args:
        file_path: Path to the file
        require_mp4: If True, only accept MP4 format (reject TS, MKV, WebM)
    """
    if not file_path.exists():
        return False
    
    file_size = file_path.stat().st_size
    if file_size == 0:
        return False
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(20)
            # MP4: starts with ftyp
            if require_mp4:
                return b'ftyp' in header  # Only accept MP4
            else:
                # TS: starts with 0x47 (sync byte) - MPEG transport stream
                # MKV/WebM: starts with 0x1A45DFA3
                return (
                    b'ftyp' in header or  # MP4
                    header[0] == 0x47 or  # TS format
                    header[:4] == b'\x1a\x45\xdf\xa3'  # MKV/WebM
                )
    except:
        return False

def find_downloaded_file(video_id, output_dir, check_valid=True, require_mp4=False):
    """Find downloaded video file with any extension
    
    Args:
        video_id: YouTube video ID
        output_dir: Output directory
        check_valid: Whether to validate the file
        require_mp4: If True, only accept MP4 format (reject TS, MKV, WebM)
    """
    # Prefer MP4 format
    for ext in ['.mp4', '.mkv', '.webm', '.ts']:
        check_file = output_dir / f"{video_id}{ext}"
        if check_file.exists():
            if check_valid:
                if is_valid_video_file(check_file, require_mp4=require_mp4):
                    return check_file
            else:
                return check_file
    return None

def cleanup_invalid_files(video_id, output_dir):
    """Remove invalid or corrupted video files"""
    cleaned = []
    for ext in ['.mp4', '.mkv', '.webm', '.ts', '.part', '.ytdl']:
        check_file = output_dir / f"{video_id}{ext}"
        if check_file.exists():
            # Check if it's a valid video file
            if ext in ['.mp4', '.mkv', '.webm', '.ts']:
                if not is_valid_video_file(check_file):
                    try:
                        check_file.unlink()
                        cleaned.append(str(check_file))
                    except Exception as e:
                        print(f"[WARNING] Could not delete invalid file {check_file}: {e}")
            else:
                # Remove temporary files (.part, .ytdl)
                try:
                    check_file.unlink()
                    cleaned.append(str(check_file))
                except Exception as e:
                    print(f"[WARNING] Could not delete temp file {check_file}: {e}")
    return cleaned

def download_video(video_id, output_dir, python_exe, youtube_dl_path, cookies_file):
    """Download a single video"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Clean up any invalid or temporary files first
    cleaned = cleanup_invalid_files(video_id, output_dir)
    if cleaned:
        print(f"[CLEANUP] Removed invalid/temp files: {', '.join(cleaned)}")
    
    # Check if video already exists (any format)
    existing_file = find_downloaded_file(video_id, output_dir)
    if existing_file:
        file_size = existing_file.stat().st_size
        format_name = existing_file.suffix[1:].upper() if existing_file.suffix else "UNKNOWN"
        print(f"[SKIP] Video already exists: {video_id} ({file_size / 1024 / 1024:.2f} MB, {format_name})")
        return True
    
    # Build command (exactly like prep/download.py)
    output_template = str(output_dir / "%(id)s.%(ext)s")
    
    if python_exe:
        # Use Python module
        # Force MP4 format: prefer mp4 containers, avoid TS format
        # Use format selector that avoids TS streams
        cmd = [
            python_exe, '-m', 'yt_dlp',
            url,
            '-f', 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '--no-check-certificate',
            '--restrict-filenames',
            '--prefer-free-formats',
            '--recode-video', 'mp4',  # Force recode to MP4 if needed
            '-o', output_template
        ]
    else:
        # Use direct command
        cmd = [
            youtube_dl_path,
            url,
            '-f', 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '--no-check-certificate',
            '--restrict-filenames',
            '--prefer-free-formats',
            '--recode-video', 'mp4',  # Force recode to MP4 if needed
            '-o', output_template
        ]
    
    # Set LD_LIBRARY_PATH for ffmpeg to find libiconv in conda environment
    env = os.environ.copy()
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        lib_path = os.path.join(conda_prefix, 'lib')
        current_ld_path = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path
    
    # Add cookies if available
    if cookies_file and cookies_file.exists():
        cmd.extend(['--cookies', str(cookies_file)])
    
    # Execute command (use list form to avoid shell quoting issues)
    try:
        result = subprocess.run(
            cmd,
            shell=False,  # Use list form to avoid shell quoting issues with brackets
            timeout=600,
            capture_output=True,
            text=True,
            env=env
        )
        
        # Check return code
        if result.returncode != 0:
            error_msg = result.stderr[:300] if result.stderr else "Unknown error"
            print(f"[ERROR] Download failed: {error_msg}")
            return False
        
        # Wait a bit for file to be written
        time.sleep(2)
        
        # Check result - look for downloaded file
        # Prefer MP4 format, but accept other formats if MP4 not available
        downloaded_file = find_downloaded_file(video_id, output_dir, check_valid=True, require_mp4=True)
        if not downloaded_file:
            # Check if we got a non-MP4 format (TS, MKV, etc.)
            downloaded_file = find_downloaded_file(video_id, output_dir, check_valid=True, require_mp4=False)
            if downloaded_file and downloaded_file.suffix != '.mp4':
                # Got a non-MP4 format - warn but accept it
                file_size = downloaded_file.stat().st_size
                format_name = downloaded_file.suffix[1:].upper()
                print(f"[WARNING] Downloaded {format_name} format instead of MP4: {video_id} ({file_size / 1024 / 1024:.2f} MB)")
                print(f"[INFO] File: {downloaded_file.name}")
                return True
        
        if downloaded_file:
            file_size = downloaded_file.stat().st_size
            format_name = downloaded_file.suffix[1:].upper() if downloaded_file.suffix else "UNKNOWN"
            print(f"[OK] Download completed: {video_id} ({file_size / 1024 / 1024:.2f} MB, {format_name})")
            return True
        else:
            print(f"[ERROR] File not found or invalid after download: {video_id}")
            # Check for any temporary files that might indicate a failed download
            temp_files = list(output_dir.glob(f"{video_id}.*"))
            if temp_files:
                print(f"[INFO] Found temporary files: {[str(f) for f in temp_files]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Download timeout: {video_id}")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading {video_id}: {e}")
        return False

def main():
    """Main function to download all videos"""
    # Get yt-dlp path
    try:
        result = subprocess.run([sys.executable, '-m', 'yt_dlp', '--version'],
                              capture_output=True, check=True, timeout=5)
        youtube_dl_path = f'PYTHON_MODULE:{sys.executable}'
        python_exe = sys.executable
        print(f"Using: {python_exe} -m yt_dlp")
    except:
        youtube_dl_path = 'yt-dlp'
        python_exe = None
        print(f"Using: yt-dlp")
    
    # Read video IDs from file
    script_dir = Path(__file__).parent
    
    # Get input file from environment variable or use default
    input_file = os.getenv('YOUTUBE_DOWNLOAD_INPUT_FILE')
    if input_file:
        video_ids_file = Path(input_file)
    else:
        video_ids_file = script_dir / "youtube_video_ids_stage2_notstrict.txt"
    
    if not video_ids_file.exists():
        print(f"[ERROR] Video IDs file not found: {video_ids_file}")
        return
    
    print(f"Reading video IDs from: {video_ids_file}")
    video_ids = read_video_ids(video_ids_file)
    print(f"Found {len(video_ids)} video IDs")
    
    # Get output directory from environment variable or use default
    output_dir_str = os.getenv('YOUTUBE_DOWNLOAD_OUTPUT_DIR')
    if output_dir_str:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path("/shared/rc/llm-gen-agent/mhu/videos/youtube_asl/downloads")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get cookies file (check in script directory first, then parent)
    cookies_file = script_dir / "www.youtube.com_cookies.txt"
    if not cookies_file.exists():
        cookies_file = script_dir.parent / "www.youtube.com_cookies.txt"
    
    if cookies_file.exists():
        print(f"Using cookies: {cookies_file}")
    else:
        print("No cookies file found (will proceed without cookies)")
        cookies_file = None
    
    print("\n" + "=" * 60)
    print("Starting downloads...")
    print("=" * 60 + "\n")
    
    # Download videos
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"\n[{i}/{len(video_ids)}] Processing: {video_id}")
        
        # Check if already exists and valid
        # Prefer MP4 format, but if only TS/MKV exists, offer to re-download
        existing_file = find_downloaded_file(video_id, output_dir, require_mp4=True)
        if not existing_file:
            # Check if we have a non-MP4 format
            existing_file = find_downloaded_file(video_id, output_dir, require_mp4=False)
            if existing_file and existing_file.suffix != '.mp4':
                # We have a non-MP4 file - delete it and re-download as MP4
                format_name = existing_file.suffix[1:].upper()
                print(f"  [INFO] Found {format_name} format, will re-download as MP4")
                try:
                    existing_file.unlink()
                    print(f"  [CLEANUP] Deleted {format_name} file: {existing_file.name}")
                except Exception as e:
                    print(f"  [WARNING] Could not delete {format_name} file: {e}")
                existing_file = None
        
        if existing_file:
            skip_count += 1
            file_size = existing_file.stat().st_size
            format_name = existing_file.suffix[1:].upper() if existing_file.suffix else "UNKNOWN"
            print(f"  [SKIP] Already downloaded and valid ({format_name}, {file_size / 1024 / 1024:.2f} MB)")
            continue
        
        # Clean up any invalid files before attempting download
        cleaned = cleanup_invalid_files(video_id, output_dir)
        if cleaned:
            print(f"  [CLEANUP] Removed invalid/temp files: {', '.join([Path(f).name for f in cleaned])}")
        
        # Download video
        result = download_video(video_id, output_dir, python_exe, youtube_dl_path, cookies_file)
        
        if result:
            success_count += 1
        else:
            fail_count += 1
        
        # Delay between downloads to avoid rate limiting
        if i < len(video_ids):
            time.sleep(2)
        
        # Print progress every 10 videos
        if i % 10 == 0:
            print(f"\nProgress: {i}/{len(video_ids)} | Success: {success_count} | Failed: {fail_count} | Skipped: {skip_count}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  Total: {len(video_ids)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Skipped: {skip_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()

