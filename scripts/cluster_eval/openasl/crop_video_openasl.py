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

def sanitize_filename(filename):
    # Replace invalid Windows characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def sample_frames(frames, original_fps, target_fps):
    """
    Sample frames based on FPS ratio.
    If target_fps < original_fps: downsample (drop frames uniformly)
    If target_fps > original_fps: keep all frames (will be duplicated by ffmpeg)
    If target_fps == original_fps: keep all frames
    
    Example: 24fps -> 12fps: ratio=2, take every 2nd frame
    """
    if target_fps is None or target_fps >= original_fps:
        return frames  # Keep all frames, ffmpeg will handle upsampling
    
    # Downsample: calculate frame skip ratio
    ratio = original_fps / target_fps
    sampled_frames = []
    # Uniformly sample: take every nth frame
    for i in range(0, len(frames), int(round(ratio))):
        sampled_frames.append(frames[i])
    
    return sampled_frames

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

def check_nvenc_support(ffmpeg):
    """Check if NVENC is supported"""
    try:
        cmd = [ffmpeg, '-encoders']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        output = result.stdout.decode('utf-8', errors='ignore') + result.stderr.decode('utf-8', errors='ignore')
        return 'h264_nvenc' in output
    except:
        return False

def write_video_ffmpeg(rois, target_path, ffmpeg, fps=25, use_gpu=False):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    tmp_dir = tempfile.mkdtemp()
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
        
        # Check if NVENC is available, otherwise use CPU encoder
        if use_gpu and check_nvenc_support(ffmpeg):
            # Use GPU encoding (NVENC)
            cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-c:v', 'h264_nvenc', '-preset', 'p1', '-rc', 'vbr', '-crf', '20', '-pix_fmt', 'yuv420p', target_path]
        else:
            # Use CPU encoding (libx264) as fallback
            cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', target_path]
        
        pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        if pipe.returncode != 0:
            print(f"ERROR: ffmpeg failed for {target_path}")
            print(pipe.stdout.decode('utf-8', errors='ignore'))
    finally:
        # Always cleanup tmp dir, even if there's an error
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp directory {tmp_dir}: {e}")
    return

def process_item_worker(args_tuple):
    """
    Worker function for multiprocessing. Must be at module level for Windows compatibility.
    args_tuple: (vid, yid, start, end, bbox, output_dir, raw_dir, ffmpeg, target_size, target_fps)
    """
    vid, yid, start, end, bbox, output_dir, raw_dir, ffmpeg_path, target_size, target_fps = args_tuple
    
    output_video = os.path.join(output_dir, sanitize_filename(vid)+'.mp4')
    if os.path.isfile(output_video):
        return True
    
    input_video_whole = os.path.join(raw_dir, yid+'.mp4')
    if not os.path.isfile(input_video_whole):
        return False
    
    # Process one clip (reuse existing code from get_clip)
    tmp_dir = tempfile.mkdtemp()
    input_video_clip = os.path.join(tmp_dir, 'tmp.mp4')
    
    # Use CPU encoding (libx264) for temporary clip - simpler and more compatible
    cmd = [ffmpeg_path, '-ss', start, '-to', end, '-i', input_video_whole, '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '1', input_video_clip]
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    cap = cv2.VideoCapture(input_video_clip)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video clip {input_video_clip}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    
    # Get original FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or math.isnan(original_fps):
        original_fps = 25.0  # Fallback to 25 if FPS detection fails
    # Use target_fps if specified, otherwise preserve original
    output_fps = target_fps if target_fps is not None else original_fps
    frames_origin = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_origin.append(frame)
    cap.release()
    time.sleep(0.1)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Check if we got any frames
    if len(frames_origin) == 0:
        print(f"ERROR: No frames extracted from {yid} (clip: {start} to {end})")
        return False
    
    # Sample frames if downsampling (target_fps < original_fps)
    if target_fps is not None and target_fps < original_fps:
        frames_origin = sample_frames(frames_origin, original_fps, target_fps)
        if len(frames_origin) == 0:
            print(f"ERROR: No frames after downsampling from {yid}")
            return False
    
    # If bbox is None, skip this video (don't process without bbox)
    if bbox is None:
        return False
    
    x0, y0, x1, y1 = bbox
    W, H = frames_origin[0].shape[1], frames_origin[0].shape[0]
    bbox = [int(x0*W), int(y0*H), int(x1*W), int(y1*H)]
    rois = crop_resize(frames_origin, bbox, target_size)
    write_video_ffmpeg(rois, output_video, ffmpeg=ffmpeg_path, fps=output_fps)
    
    return os.path.isfile(output_video)

def get_clip(input_video_dir, output_video_dir, tsv_fn, bbox_fn, rank, nshard, target_size=224, ffmpeg=None, target_fps=None):
    os.makedirs(output_video_dir, exist_ok=True)
    df = pd.read_csv(tsv_fn, sep='\t')
    
    # Load bbox - required for processing
    vid2bbox = {}
    if bbox_fn and os.path.isfile(bbox_fn):
        vid2bbox = json.load(open(bbox_fn))
    elif bbox_fn:
        print(f"WARNING: Bbox file specified but not found: {bbox_fn}")
        print("Only videos with bbox will be processed. If bbox file is missing, videos will be skipped.")
    
    items = []
    # Only process videos that have bbox - skip videos without bbox
    for vid, yid, start, end in zip(df['vid'], df['yid'], df['start'], df['end']):
        if vid not in vid2bbox:
            continue  # Skip videos without bbox
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
        cmd = [ffmpeg, '-ss', start_time, '-to', end_time, '-i', input_video_whole, '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '1', input_video_clip]
        print(' '.join(cmd))
        subprocess.call(cmd)
        cap = cv2.VideoCapture(input_video_clip)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video clip {input_video_clip}")
            import time
            time.sleep(0.1)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
        
        # Get original FPS
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0 or math.isnan(original_fps):
            original_fps = 25.0  # Fallback to 25 if FPS detection fails
        # Use target_fps if specified, otherwise preserve original
        output_fps = target_fps if target_fps is not None else original_fps
        frames_origin = []
        print(f"Reading video clip: {input_video_clip} (Original FPS: {original_fps:.2f}, Output FPS: {output_fps:.2f})")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_origin.append(frame)
        cap.release()  # Release the video capture before deleting the file
        import time
        time.sleep(0.1)  # Give Windows time to release the file handle
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
        # Check if we got any frames
        if len(frames_origin) == 0:
            print(f"ERROR: No frames extracted from {yid} (clip: {start_time} to {end_time})")
            continue
        
        # Sample frames if downsampling (target_fps < original_fps)
        if target_fps is not None and target_fps < original_fps:
            print(f"Downsampling: {len(frames_origin)} frames -> ", end="")
            frames_origin = sample_frames(frames_origin, original_fps, target_fps)
            print(f"{len(frames_origin)} frames")
            if len(frames_origin) == 0:
                print(f"ERROR: No frames after downsampling from {yid}")
                continue
        
        # If bbox is None, skip this video (don't process without bbox)
        if bbox is None:
            print(f"Skipping {vid}: no bbox found")
            continue
        
        x0, y0, x1, y1 = bbox
        W, H = frames_origin[0].shape[1], frames_origin[0].shape[0]
        bbox = [int(x0*W), int(y0*H), int(x1*W), int(y1*H)]
        print(bbox, frames_origin[0].shape, target_size)
        rois = crop_resize(frames_origin, bbox, target_size)
        print(f"Saving ROIs to {output_video}")
        write_video_ffmpeg(rois, output_video, ffmpeg=ffmpeg, fps=output_fps)
        if os.path.isfile(output_video):
            print(f"✓ Successfully saved {output_video}")
        else:
            print(f"✗ ERROR: File not created: {output_video}")
    return


def main():
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_tsv = os.path.join(script_dir, 'openasl-no-nad.tsv')
    default_bbox = os.path.join(script_dir, 'bbox-v1.0.json')
    
    parser = argparse.ArgumentParser(description='download video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, default=default_tsv, help='data tsv file')
    parser.add_argument('--bbox', type=str, default=default_bbox, help='bbox json file (optional - if not provided, will use entire video frame)')
    parser.add_argument('--raw', type=str, default='/shared/rc/llm-gen-agent/mhu/videos/open_asl/raw_videos', help='raw video dir')
    parser.add_argument('--output', type=str, help='output dir')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='path to ffmpeg')
    parser.add_argument('--target-size', type=int, default=224, help='image size')
    parser.add_argument('--target-fps', type=float, default=25.0, help='target FPS to normalize all videos (default: 25.0)')

    parser.add_argument('--slurm', action='store_true', help='slurm or not')
    parser.add_argument('--nshard', type=int, default=100, help='number of slurm jobs to launch in total')
    parser.add_argument('--workers', type=int, default=None, help='number of parallel workers (None/not set = single thread, 0 = use all CPU cores, N = use N workers)')
    parser.add_argument('--slurm-argument', type=str, default='{"slurm_array_parallelism":100,"slurm_partition":"speech-cpu","timeout_min":240,"slurm_mem":"16g"}', help='slurm arguments')
    args = parser.parse_args()

    # Auto-detect ffmpeg on Windows if not specified
    if args.ffmpeg == 'ffmpeg' and os.name == 'nt':
        # Look for ffmpeg.exe in the project root
        ffmpeg_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffmpeg.exe")
        if os.path.isfile(ffmpeg_exe):
            args.ffmpeg = ffmpeg_exe
        else:
            args.ffmpeg = "ffmpeg.exe"

    if args.slurm:
        import submitit
        nshard = args.nshard
        executor = submitit.AutoExecutor(folder='submitit')
        params = json.loads(args.slurm_argument)
        executor.update_parameters(**params)
        bbox_arg = args.bbox if args.bbox else ''
        jobs = executor.map_array(get_clip, [args.raw]*nshard, [args.output]*nshard, [args.tsv]*nshard, [bbox_arg]*nshard, list(range(0, nshard)), [nshard]*nshard, [args.target_size]*nshard, [args.ffmpeg]*nshard, [args.target_fps]*nshard)
    elif args.workers is not None:
        # Multiprocessing mode
        # workers=0 means use all CPU cores, workers>0 means use that many workers
        cpu_count = mp.cpu_count()
        if args.workers == 0:
            # Use all CPU cores for maximum parallelism (GPU encoding doesn't compete for CPU)
            num_workers = cpu_count
        else:
            num_workers = args.workers
        print(f"Using {num_workers} parallel workers (CPU cores: {cpu_count})")
        
        # Load data once
        df = pd.read_csv(args.tsv, sep='\t', low_memory=False)
        
        # Load bbox - required for processing
        vid2bbox = {}
        if args.bbox and os.path.isfile(args.bbox):
            vid2bbox = json.load(open(args.bbox))
        elif args.bbox:
            print(f"WARNING: Bbox file specified but not found: {args.bbox}")
            print("Only videos with bbox will be processed. If bbox file is missing, videos will be skipped.")
        
        # Only process videos that have bbox - skip videos without bbox
        items = []
        for vid, yid, start, end in zip(df['vid'], df['yid'], df['start'], df['end']):
            if vid not in vid2bbox:
                continue  # Skip videos without bbox
            bbox = vid2bbox[vid]
            items.append([vid, yid, start, end, bbox])
        
        # Prepare items with all necessary arguments for the worker function
        # Each item: (vid, yid, start, end, bbox, output_dir, raw_dir, ffmpeg, target_size, target_fps)
        worker_items = [
            (vid, yid, start, end, bbox, args.output, args.raw, args.ffmpeg, args.target_size, args.target_fps)
            for vid, yid, start, end, bbox in items
        ]
        
        # Process in parallel
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_item_worker, worker_items), total=len(worker_items)))
        print(f"Successfully processed {sum(results)} videos")
    else:
        bbox_arg = args.bbox if args.bbox else ''
        get_clip(args.raw, args.output, args.tsv, bbox_arg, 0, 1, target_size=args.target_size, ffmpeg=args.ffmpeg, target_fps=args.target_fps)
    return


if __name__ == '__main__':
    main()
