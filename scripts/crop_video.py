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

def resolve_ffmpeg_path(ffmpeg_arg):
    """Resolve the ffmpeg executable path to an absolute path.
    This is important for multiprocessing workers which may not have the same PATH."""
    # If it's already an absolute path and exists, use it
    if os.path.isabs(ffmpeg_arg) and os.path.isfile(ffmpeg_arg):
        return os.path.abspath(ffmpeg_arg)
    
    # If it's a relative path and exists, make it absolute
    if os.path.isfile(ffmpeg_arg):
        return os.path.abspath(ffmpeg_arg)
    
    # Try to find it in PATH
    ffmpeg_path = shutil.which(ffmpeg_arg)
    if ffmpeg_path:
        return os.path.abspath(ffmpeg_path)
    
    # On Windows, try to find ffmpeg.exe in project root
    if os.name == 'nt' and ffmpeg_arg in ['ffmpeg', 'ffmpeg.exe']:
        ffmpeg_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffmpeg.exe")
        if os.path.isfile(ffmpeg_exe):
            return os.path.abspath(ffmpeg_exe)
    
    # If not found, return the original (will fail later with a clearer error)
    # But still try to make it absolute if it looks like a path
    if os.path.sep in ffmpeg_arg:
        return os.path.abspath(ffmpeg_arg)
    return ffmpeg_arg

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

def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 30
    tmp_dir = tempfile.mkdtemp()
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
    # rm tmp dir
    shutil.rmtree(tmp_dir)
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
    default_name = inp.stem + f"_crop_{target_size}.mp4"
    if out_path is None:
        out_path = str(inp.with_name(default_name))
    else:
        out_path_path = Path(out_path)
        if out_path_path.is_dir() or out_path.endswith(os.sep):
            out_path_path.mkdir(parents=True, exist_ok=True)
            out_path = str(out_path_path / default_name)
        else:
            out_path_path.parent.mkdir(parents=True, exist_ok=True)
    write_video_ffmpeg(rois, out_path, ffmpeg)
    if not os.path.isfile(out_path):
        raise RuntimeError(f"Failed to create output video: {out_path}")
    # Use tqdm.write to avoid interfering with progress bars
    try:
        tqdm.write(f"✓ Saved cropped video to {out_path}")
    except:
        print(f"✓ Saved cropped video to {out_path}")
    return

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
        tmp_dir = tempfile.mkdtemp()
        input_video_clip = os.path.join(tmp_dir, 'tmp.mp4')
        cmd = [ffmpeg, '-ss', start_time, '-to', end_time, '-i', input_video_whole, '-c:v', 'libx264', '-crf', '20', input_video_clip]
        print(' '.join(cmd))
        subprocess.call(cmd)
        cap = cv2.VideoCapture(input_video_clip)
        frames_origin = []
        print(f"Reading video clip: {input_video_clip}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_origin.append(frame)
        cap.release()  # Release the video capture before deleting the file
        import time
        time.sleep(0.1)  # Give Windows time to release the file handle
        shutil.rmtree(tmp_dir)
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
        write_video_ffmpeg(rois, output_video, ffmpeg=ffmpeg)
        if os.path.isfile(output_video):
            print(f"✓ Successfully saved {output_video}")
        else:
            print(f"✗ ERROR: File not created: {output_video}")
    return

def process_directory(input_dir, output_dir, bbox, target_size, ffmpeg, bbox_scale, num_workers=1):
    """Process all videos in a directory with the same bbox coordinates."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    
    # Find all video files in the directory
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        video_files.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))
    
    if not video_files:
        raise ValueError(f"No video files found in directory: {input_dir}")
    
    print(f"Found {len(video_files)} video files in {input_dir}")
    
    # Ensure output directory exists
    # Find the deepest existing parent directory
    output_path = Path(output_dir)
    existing_parent = output_path
    while existing_parent != existing_parent.parent and not existing_parent.exists():
        existing_parent = existing_parent.parent
    
    if not existing_parent.exists():
        raise NotADirectoryError(f"None of the parent directories exist for: {output_dir}")
    
    if not os.access(existing_parent, os.W_OK):
        raise PermissionError(f"No write permission for existing parent directory: {existing_parent}. Cannot create: {output_dir}")
    
    # Try to create the output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create output directory {output_dir}. Error: {e}. "
                            f"Deepest existing parent: {existing_parent}")
    except OSError as e:
        raise OSError(f"Cannot create output directory {output_dir}. Error: {e}. "
                     f"Deepest existing parent: {existing_parent}")
    
    # Prepare items for processing
    items = []
    for input_video in video_files:
        inp = Path(input_video)
        output_filename = inp.stem + f"_crop_{target_size}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        items.append((input_video, output_path))
    
    # Process videos
    if num_workers > 1:
        # Parallel processing
        print(f"Using {num_workers} parallel workers")
        def process_item(args_tuple):
            input_video, output_path = args_tuple
            inp = Path(input_video)
            try:
                # Skip if output already exists
                if os.path.isfile(output_path):
                    return True, f"Skipped {inp.name}"
                
                # Process the video
                process_single_video(input_video, output_path, bbox, target_size, ffmpeg, bbox_scale)
                return True, f"Processed {inp.name}"
            except Exception as e:
                return False, f"Error processing {inp.name}: {str(e)}"
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_item, items), total=len(items), desc="Processing videos"))
        
        successful = sum(1 for success, _ in results if success)
        print(f"✓ Finished processing: {successful}/{len(items)} videos successful")
    else:
        # Sequential processing with better progress reporting
        for i, (input_video, output_path) in enumerate(tqdm(items, desc="Processing videos"), 1):
            try:
                inp = Path(input_video)
                # Skip if output already exists
                if os.path.isfile(output_path):
                    tqdm.write(f"[{i}/{len(items)}] ⏭ Skipping {inp.name} (output already exists)")
                    continue
                
                tqdm.write(f"[{i}/{len(items)}] Processing {inp.name}...")
                # Process the video
                process_single_video(input_video, output_path, bbox, target_size, ffmpeg, bbox_scale)
            except Exception as e:
                tqdm.write(f"[{i}/{len(items)}] ✗ Error processing {input_video}: {str(e)}")
                continue
        
        print(f"✓ Finished processing {len(items)} videos")
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
    parser.add_argument('--input-dir', type=str, help='Process all videos in a directory (overrides TSV mode)')
    parser.add_argument('--output-video', type=str, help='Output path for single video or output directory for batch processing')
    parser.add_argument('--bbox-coords', type=str, help='Bounding box as x0,y0,x1,y1 (normalized 0-1 or pixels) for single video or batch processing')
    parser.add_argument('--bbox-scale', type=float, default=0.85, help='Scale factor (0-1) to shrink bbox around center for tighter crop (default: 0.85)')

    parser.add_argument('--slurm', action='store_true', help='slurm or not')
    parser.add_argument('--nshard', type=int, default=100, help='number of slurm jobs to launch in total')
    parser.add_argument('--workers', type=int, default=0, help='number of parallel workers (0 = use all CPU cores)')
    parser.add_argument('--slurm-argument', type=str, default='{"slurm_array_parallelism":100,"slurm_partition":"speech-cpu","timeout_min":240,"slurm_mem":"16g"}', help='slurm arguments')
    args = parser.parse_args()

    # Resolve ffmpeg path (important for multiprocessing workers)
    args.ffmpeg = resolve_ffmpeg_path(args.ffmpeg)
    
    # Verify ffmpeg is accessible
    if not os.path.isfile(args.ffmpeg):
        # Try to run it to see if it's in PATH
        try:
            result = subprocess.run([args.ffmpeg, '-version'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  timeout=5)
            if result.returncode != 0 and 'ffmpeg' not in result.stderr.decode('utf-8', errors='ignore').lower():
                raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise FileNotFoundError(f"ffmpeg not found or not executable: {args.ffmpeg}. "
                                  f"Make sure ffmpeg is installed and in PATH. Error: {e}")

    if args.input_dir:
        if args.bbox_coords is None:
            raise ValueError("--bbox-coords is required when processing a directory")
        if args.output_video is None:
            raise ValueError("--output-video is required when processing a directory (specify output directory)")
        bbox_list = [v.strip() for v in args.bbox_coords.replace(' ', '').split(',')]
        if len(bbox_list) != 4:
            raise ValueError("bbox-coords must have four values: x0,y0,x1,y1")
        # Ensure output_video is treated as a directory
        output_dir = args.output_video
        if not output_dir.endswith(os.sep) and not os.path.isdir(output_dir):
            # If it doesn't exist and doesn't end with separator, treat as directory
            pass
        # Use workers if specified, otherwise default to 1 (sequential)
        num_workers = args.workers if args.workers > 0 else 1
        process_directory(args.input_dir, output_dir, bbox_list, args.target_size, args.ffmpeg, args.bbox_scale, num_workers)
        return
    
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
        from tqdm import tqdm
        def process_item(args_tuple):
            vid, yid, start, end, bbox = args_tuple
            
            output_video = os.path.join(args.output, sanitize_filename(vid)+'.mp4')
            if os.path.isfile(output_video):
                return True
            
            input_video_whole = os.path.join(args.raw, yid+'.mp4')
            if not os.path.isfile(input_video_whole):
                return False
            
            # Process one clip (reuse existing code from get_clip)
            tmp_dir = tempfile.mkdtemp()
            input_video_clip = os.path.join(tmp_dir, 'tmp.mp4')
            
            # Use GPU encoding (NVENC) for much faster processing
            cmd = [args.ffmpeg, '-ss', start, '-to', end, '-i', input_video_whole, '-c:v', 'h264_nvenc', '-preset', 'p1', '-crf', '20', '-rc', 'vbr', input_video_clip]
            subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            cap = cv2.VideoCapture(input_video_clip)
            frames_origin = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_origin.append(frame)
            cap.release()
            time.sleep(0.1)
            shutil.rmtree(tmp_dir)
            
            x0, y0, x1, y1 = bbox
            W, H = frames_origin[0].shape[1], frames_origin[0].shape[0]
            x0p = x0 * W if max(bbox) <= 1.0 else x0
            y0p = y0 * H if max(bbox) <= 1.0 else y0
            x1p = x1 * W if max(bbox) <= 1.0 else x1
            y1p = y1 * H if max(bbox) <= 1.0 else y1
            x0p, y0p, x1p, y1p = adjust_bbox_with_margin(x0p, y0p, x1p, y1p, W, H, args.bbox_scale)
            rois = crop_resize(frames_origin, [x0p, y0p, x1p, y1p], args.target_size)
            write_video_ffmpeg(rois, output_video, ffmpeg=args.ffmpeg)
            
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
