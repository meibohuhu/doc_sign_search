import os
import json
import math
import cv2
import sys
import glob
import subprocess
import shutil
import tempfile
import argparse
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from pathlib import Path

def sanitize_filename(filename):
    # Replace invalid Windows characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def cleanup_old_temp_dirs(temp_base='/local1/tmp', max_age_hours=24):
    """
    Clean up old temporary directories that are older than max_age_hours.
    
    Args:
        temp_base: Base directory for temporary files
        max_age_hours: Maximum age in hours for temp directories (default: 24)
    """
    if not os.path.exists(temp_base):
        return
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    cleaned_count = 0
    cleaned_size = 0
    
    try:
        for item in os.listdir(temp_base):
            item_path = os.path.join(temp_base, item)
            if not os.path.isdir(item_path):
                continue
            
            # Check if it's a temp directory (starts with 'tmp')
            if not item.startswith('tmp'):
                continue
            
            try:
                # Get modification time
                mtime = os.path.getmtime(item_path)
                age_seconds = current_time - mtime
                
                if age_seconds > max_age_seconds:
                    # Calculate size before deletion
                    try:
                        size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                   for dirpath, dirnames, filenames in os.walk(item_path)
                                   for filename in filenames)
                        cleaned_size += size
                    except:
                        pass
                    
                    # Remove the directory
                    shutil.rmtree(item_path)
                    cleaned_count += 1
            except Exception as e:
                # Skip directories that can't be accessed or deleted
                continue
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old temp directories ({cleaned_size / (1024**3):.2f} GB)")
    except Exception as e:
        print(f"Warning: Error during temp directory cleanup: {e}")

def crop_resize(imgs, bbox, target_size):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    if x1-x0<y1-y0:
        exp = (y1-y0-(x1-x0))/2
        x0, x1 = x0-exp, x1+exp
    else:
        exp = (x1-x0-(y1-y0))/2
        y0, y1 = y0-exp, y1+exp
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    left_expand = -x0 if x0 < 0 else 0
    up_expand = -y0 if y0 < 0 else 0
    right_expand = x1-imgs[0].shape[1]+1 if x1 > imgs[0].shape[1]-1 else 0
    down_expand = y1-imgs[0].shape[0]+1 if y1 > imgs[0].shape[0]-1 else 0
    rois = []
    for img in imgs:
        expand_img = cv2.copyMakeBorder(img, up_expand, down_expand, left_expand, right_expand, cv2.BORDER_CONSTANT, (0, 0, 0))
        roi = expand_img[y0+up_expand: y1+up_expand, x0+left_expand: x1+left_expand]
        roi = cv2.resize(roi, (target_size, target_size))
        rois.append(roi)
    return rois

def write_video_ffmpeg(rois, target_path, ffmpeg, fps=None):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    if fps is None:
        fps = 30  # Default FPS if not specified
    # Use /local1 for temp directory if available (to avoid root partition space issues)
    temp_base = '/local1/tmp' if os.path.exists('/local1') else None
    if temp_base:
        os.makedirs(temp_base, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=temp_base)
    try:
        for i_roi, roi in enumerate(rois):
            cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
        list_fn = os.path.join(tmp_dir, "list")
        with open(list_fn, 'w') as fo:
            # Use forward slashes for ffmpeg (works on both Windows and Unix)
            pattern = os.path.join(tmp_dir, f'%0{decimals}d.png').replace('\\', '/')
            fo.write(f"file '{pattern}'\n")
        ## ffmpeg
        if os.path.isfile(target_path):
            os.remove(target_path)
        cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-c:v', 'h264_nvenc', '-preset', 'p1', '-rc', 'vbr', '-crf', '20', '-pix_fmt', 'yuv420p', target_path]
        pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        if pipe.returncode != 0:
            print(f"ERROR: ffmpeg failed for {target_path}")
            print(pipe.stdout.decode('utf-8', errors='ignore'))
    finally:
        # Always cleanup tmp dir, even if there's an error
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            # Log but don't fail if cleanup fails
            print(f"Warning: Failed to cleanup temp directory {tmp_dir}: {e}")
    return

def adjust_bbox_with_margin(x0, y0, x1, y1, frame_w, frame_h, scale):
    if scale >= 1.0 or scale <= 0:
        return x0, y0, x1, y1
    width = x1 - x0
    height = y1 - y0
    cx = x0 + width / 2.0
    cy = y0 + height / 2.0
    half_w = width * scale / 2.0
    half_h = height * scale / 2.0
    new_x0 = max(0, cx - half_w)
    new_x1 = min(frame_w, cx + half_w)
    new_y0 = max(0, cy - half_h)
    new_y1 = min(frame_h, cy + half_h)
    return new_x0, new_y0, new_x1, new_y1

def process_single_video(input_video, output_video, bbox, target_size, ffmpeg, bbox_scale):
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    cap = cv2.VideoCapture(input_video)
    
    # Get original FPS from input video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or original_fps > 120:
        # Fallback to default if FPS is invalid
        original_fps = 30
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Could not read any frames from {input_video}")
    h, w = frames[0].shape[:2]
    bbox_vals = [float(v) for v in bbox]
    if max(bbox_vals) <= 1.0:
        x0 = bbox_vals[0] * w
        y0 = bbox_vals[1] * h
        x1 = bbox_vals[2] * w
        y1 = bbox_vals[3] * h
    else:
        x0, y0, x1, y1 = bbox_vals
    x0, y0, x1, y1 = adjust_bbox_with_margin(x0, y0, x1, y1, w, h, bbox_scale)
    bbox_pixels = [int(x0), int(y0), int(x1), int(y1)]
    rois = crop_resize(frames, bbox_pixels, target_size)
    out_path = output_video
    inp = Path(input_video)
    default_name = inp.name  # Use original filename
    if out_path is None:
        out_path = str(inp.with_name(default_name))
    else:
        out_path_path = Path(out_path)
        if out_path_path.is_dir() or out_path.endswith(os.sep):
            out_path_path.mkdir(parents=True, exist_ok=True)
            out_path = str(out_path_path / default_name)
        else:
            out_path_path.parent.mkdir(parents=True, exist_ok=True)
    write_video_ffmpeg(rois, out_path, ffmpeg, fps=original_fps)
    if not os.path.isfile(out_path):
        raise RuntimeError(f"Failed to create output video: {out_path}")
    print(f"✓ Saved cropped video to {out_path}")
    return

def process_video_file_for_multiprocessing(video_file_path, output_dir, bbox_list, target_size, ffmpeg, bbox_scale):
    """
    Process a single video file - module-level function for multiprocessing.
    This function can be pickled by multiprocessing.
    
    Args:
        video_file_path: Path to input video file
        output_dir: Output directory path
        bbox_list: Bounding box coordinates as list
        target_size: Target output size
        ffmpeg: Path to ffmpeg executable
        bbox_scale: Bbox scale factor
        
    Returns:
        dict: Status information {'status': 'success'|'skipped'|'error', 'file': filename, ...}
    """
    video_file = Path(video_file_path)
    output_video = Path(output_dir) / video_file.name  # Use original filename
    
    # Skip if output already exists
    if output_video.exists():
        return {'status': 'skipped', 'file': video_file.name}
    
    try:
        process_single_video(
            str(video_file),
            str(output_video),
            bbox_list,
            target_size,
            ffmpeg,
            bbox_scale
        )
        return {'status': 'success', 'file': video_file.name}
    except Exception as e:
        return {'status': 'error', 'file': video_file.name, 'error': str(e)}

def get_clip(input_video_dir, output_video_dir, tsv_fn, bbox_fn, rank, nshard, target_size=224, ffmpeg=None, bbox_scale=1.0):
    os.makedirs(output_video_dir, exist_ok=True)
    df = pd.read_csv(tsv_fn, sep='\t')
    vid2bbox = json.load(open(bbox_fn))
    items = []
    for vid, yid, start, end in zip(df['vid'], df['yid'], df['start'], df['end']):
        if vid not in vid2bbox:
            continue
        bbox = vid2bbox[vid]
        items.append([vid, yid, start, end, bbox])
    
    num_per_shard = (len(items)+nshard-1)//nshard
    items = items[num_per_shard*rank: num_per_shard*(rank+1)]
    print(f"{len(items)} videos")
    for vid, yid, start_time, end_time, bbox in tqdm(items):
        input_video_whole = os.path.join(input_video_dir, yid+'.mp4')
        output_video = os.path.join(output_video_dir, sanitize_filename(vid)+'.mp4')
        if os.path.isfile(output_video):
            continue
        # Use /local1 for temp directory if available (to avoid root partition space issues)
        temp_base = '/local1/tmp' if os.path.exists('/local1') else None
        if temp_base:
            os.makedirs(temp_base, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=temp_base)
        try:
            input_video_clip = os.path.join(tmp_dir, 'tmp.mp4')
            cmd = [ffmpeg, '-ss', start_time, '-to', end_time, '-i', input_video_whole, '-c:v', 'libx264', '-crf', '20', input_video_clip]
            print(' '.join(cmd))
            subprocess.call(cmd)
            cap = cv2.VideoCapture(input_video_clip)
            # Get FPS from the clip
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps <= 0 or original_fps > 120:
                original_fps = 30
            
            frames_origin = []
            print(f"Reading video clip: {input_video_clip}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_origin.append(frame)
            cap.release()  # Release the video capture before deleting the file
            time.sleep(0.1)  # Give Windows time to release the file handle
        finally:
            # Always cleanup tmp dir, even if there's an error
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory {tmp_dir}: {e}")
        x0, y0, x1, y1 = bbox
        W, H = frames_origin[0].shape[1], frames_origin[0].shape[0]
        x0p = x0 * W if max(bbox) <= 1.0 else x0
        y0p = y0 * H if max(bbox) <= 1.0 else y0
        x1p = x1 * W if max(bbox) <= 1.0 else x1
        y1p = y1 * H if max(bbox) <= 1.0 else y1
        x0p, y0p, x1p, y1p = adjust_bbox_with_margin(x0p, y0p, x1p, y1p, W, H, bbox_scale)
        print([int(x0p), int(y0p), int(x1p), int(y1p)], frames_origin[0].shape, target_size)
        rois = crop_resize(frames_origin, [x0p, y0p, x1p, y1p], target_size)
        print(f"Saving ROIs to {output_video}")
        write_video_ffmpeg(rois, output_video, ffmpeg=ffmpeg, fps=original_fps)
        if os.path.isfile(output_video):
            print(f"✓ Successfully saved {output_video}")
        else:
            print(f"✗ ERROR: File not created: {output_video}")
    return


def main():
    parser = argparse.ArgumentParser(description='download video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, help='data tsv file')
    parser.add_argument('--bbox', type=str, help='bbox json file')
    parser.add_argument('--raw', type=str, help='raw video dir')
    parser.add_argument('--output', type=str, help='output dir')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='path to ffmpeg')
    parser.add_argument('--target-size', type=int, default=720, help='output square size (default: 720)')
    parser.add_argument('--input-video', type=str, help='Process a single video (overrides TSV mode)')
    parser.add_argument('--output-video', type=str, help='Output path for single video')
    parser.add_argument('--bbox-coords', type=str, help='Bounding box as x0,y0,x1,y1 (normalized 0-1 or pixels) for single video')
    parser.add_argument('--bbox-scale', type=float, default=0.85, help='Scale factor (0-1) to shrink bbox around center for tighter crop (default: 0.85)')
    parser.add_argument('--input-dir', type=str, help='Process all videos from a directory (overrides TSV and single video mode)')
    parser.add_argument('--output-dir', type=str, help='Output directory for batch processing (required with --input-dir)')
    parser.add_argument('--default-bbox', type=str, default='0.2,0.2,0.8,0.8', help='Default bounding box as x0,y0,x1,y1 (normalized 0-1) for batch processing. Default: 0.2,0.2,0.8,0.8 (center 60 percent of frame)')

    parser.add_argument('--slurm', action='store_true', help='slurm or not')
    parser.add_argument('--nshard', type=int, default=100, help='number of slurm jobs to launch in total')
    parser.add_argument('--workers', type=int, default=0, help='number of parallel workers (0 = use all CPU cores)')
    parser.add_argument('--slurm-argument', type=str, default='{"slurm_array_parallelism":100,"slurm_partition":"speech-cpu","timeout_min":240,"slurm_mem":"16g"}', help='slurm arguments')
    parser.add_argument('--cleanup-temp', action='store_true', help='Clean up old temporary directories before processing')
    parser.add_argument('--cleanup-temp-age', type=int, default=24, help='Maximum age in hours for temp directories to keep (default: 24)')
    args = parser.parse_args()

    # Auto-detect ffmpeg on Windows if not specified
    if args.ffmpeg == 'ffmpeg' and os.name == 'nt':
        # Look for ffmpeg.exe in the project root
        ffmpeg_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffmpeg.exe")
        if os.path.isfile(ffmpeg_exe):
            args.ffmpeg = ffmpeg_exe
        else:
            args.ffmpeg = "ffmpeg.exe"

    # Cleanup old temp directories if requested
    if args.cleanup_temp:
        temp_base = '/local1/tmp' if os.path.exists('/local1') else None
        if temp_base:
            cleanup_old_temp_dirs(temp_base, args.cleanup_temp_age)

    # Batch process directory mode (highest priority)
    if args.input_dir:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when using --input-dir")
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Parse default bbox (support both --default-bbox and --bbox-coords)
        bbox_str = args.bbox_coords if args.bbox_coords else args.default_bbox
        if args.bbox_coords:
            print(f"ℹ️  Using --bbox-coords for batch processing (consider using --default-bbox in future)")
        bbox_list = [v.strip() for v in bbox_str.replace(' ', '').split(',')]
        if len(bbox_list) != 4:
            raise ValueError("bbox must have four values: x0,y0,x1,y1")
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.glob(f'*{ext}'))
            video_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"⚠️  No video files found in {input_dir}")
            return
        
        print(f"📹 Found {len(video_files)} video files")
        print(f"📁 Input directory: {input_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📐 Bbox: {bbox_str} (normalized)")
        print(f"📏 Target size: {args.target_size}x{args.target_size}")
        print(f"🔧 Bbox scale: {args.bbox_scale}")
        
        # Determine number of workers
        num_workers = args.workers if args.workers > 0 else mp.cpu_count()
        if num_workers > 1:
            print(f"🚀 Using {num_workers} parallel workers")
        else:
            print(f"🐌 Using single-threaded processing")
        print("-" * 60)
        
        # Prepare arguments for multiprocessing
        # Create a partial function with all necessary parameters
        from functools import partial
        process_func = partial(
            process_video_file_for_multiprocessing,
            output_dir=str(output_dir),
            bbox_list=bbox_list,
            target_size=args.target_size,
            ffmpeg=args.ffmpeg,
            bbox_scale=args.bbox_scale
        )
        
        # Process videos
        success_count = 0
        skipped_count = 0
        error_count = 0
        
        if num_workers > 1:
            # Multiprocessing mode
            video_paths = [str(vf) for vf in video_files]
            with mp.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, video_paths),
                    total=len(video_paths),
                    desc="Processing videos"
                ))
        else:
            # Single-threaded mode
            results = []
            for video_file in tqdm(video_files, desc="Processing videos"):
                result = process_func(str(video_file))
                results.append(result)
        
        # Count results
        for result in results:
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skipped':
                skipped_count += 1
            elif result['status'] == 'error':
                error_count += 1
                print(f"❌ Error processing {result['file']}: {result.get('error', 'Unknown error')}")
        
        print("-" * 60)
        print(f"✅ Successfully processed: {success_count} videos")
        if skipped_count > 0:
            print(f"⏭️  Skipped (already exist): {skipped_count} videos")
        if error_count > 0:
            print(f"❌ Errors: {error_count} videos")
        return
    
    # Single video mode
    if args.input_video:
        if args.bbox_coords is None:
            raise ValueError("--bbox-coords is required when processing a single video")
        bbox_list = [v.strip() for v in args.bbox_coords.replace(' ', '').split(',')]
        if len(bbox_list) != 4:
            raise ValueError("bbox-coords must have four values: x0,y0,x1,y1")
        process_single_video(args.input_video, args.output_video, bbox_list, args.target_size, args.ffmpeg, args.bbox_scale)
        return

    if args.slurm:
        import submitit
        nshard = args.nshard
        executor = submitit.AutoExecutor(folder='submitit')
        params = json.loads(args.slurm_argument)
        executor.update_parameters(**params)
        jobs = executor.map_array(get_clip, [args.raw]*nshard, [args.output]*nshard, [args.tsv]*nshard, [args.bbox]*nshard, list(range(0, nshard)), [nshard]*nshard, [args.target_size]*nshard, [args.ffmpeg]*nshard, [args.bbox_scale]*nshard)
    elif args.workers > 0:
        # Multiprocessing mode
        num_workers = args.workers if args.workers > 0 else mp.cpu_count()
        print(f"Using {num_workers} parallel workers")
        
        # Load data once
        df = pd.read_csv(args.tsv, sep='\t', low_memory=False)
        vid2bbox = json.load(open(args.bbox))
        
        # Filter items that have bounding boxes
        items = []
        for vid, yid, start, end in zip(df['vid'], df['yid'], df['start'], df['end']):
            if vid in vid2bbox:
                items.append([vid, yid, start, end, vid2bbox[vid]])
        
        # Split work across workers
        def process_item(args_tuple):
            vid, yid, start, end, bbox = args_tuple
            
            output_video = os.path.join(args.output, sanitize_filename(vid)+'.mp4')
            if os.path.isfile(output_video):
                return True
            
            input_video_whole = os.path.join(args.raw, yid+'.mp4')
            if not os.path.isfile(input_video_whole):
                return False
            
            # Process one clip (reuse existing code from get_clip)
            # Use /local1 for temp directory if available (to avoid root partition space issues)
            temp_base = '/local1/tmp' if os.path.exists('/local1') else None
            if temp_base:
                os.makedirs(temp_base, exist_ok=True)
            tmp_dir = tempfile.mkdtemp(dir=temp_base)
            try:
                input_video_clip = os.path.join(tmp_dir, 'tmp.mp4')
                
                # Use GPU encoding (NVENC) for much faster processing
                cmd = [args.ffmpeg, '-ss', start, '-to', end, '-i', input_video_whole, '-c:v', 'h264_nvenc', '-preset', 'p1', '-crf', '20', '-rc', 'vbr', input_video_clip]
                subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                cap = cv2.VideoCapture(input_video_clip)
                # Get FPS from the clip
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                if original_fps <= 0 or original_fps > 120:
                    original_fps = 30
                
                frames_origin = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_origin.append(frame)
                cap.release()
                time.sleep(0.1)
            finally:
                # Always cleanup tmp dir, even if there's an error
                try:
                    shutil.rmtree(tmp_dir)
                except Exception as e:
                    pass  # Silently ignore cleanup errors in multiprocessing
            
            x0, y0, x1, y1 = bbox
            W, H = frames_origin[0].shape[1], frames_origin[0].shape[0]
            x0p = x0 * W if max(bbox) <= 1.0 else x0
            y0p = y0 * H if max(bbox) <= 1.0 else y0
            x1p = x1 * W if max(bbox) <= 1.0 else x1
            y1p = y1 * H if max(bbox) <= 1.0 else y1
            x0p, y0p, x1p, y1p = adjust_bbox_with_margin(x0p, y0p, x1p, y1p, W, H, args.bbox_scale)
            rois = crop_resize(frames_origin, [x0p, y0p, x1p, y1p], args.target_size)
            write_video_ffmpeg(rois, output_video, ffmpeg=args.ffmpeg, fps=original_fps)
            
            return os.path.isfile(output_video)
        
        # Process in parallel
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_item, items), total=len(items)))
        print(f"Successfully processed {sum(results)} videos")
    else:
        get_clip(args.raw, args.output, args.tsv, args.bbox, 0, 1, target_size=args.target_size, ffmpeg=args.ffmpeg, bbox_scale=args.bbox_scale)
    return


if __name__ == '__main__':
    main()
