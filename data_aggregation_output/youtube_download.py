"""
Download videos from youtube_video_ids_stage2.txt

Improvements over original:
  - yt-dlp native sleep/retry options to avoid rate limits
  - Per-video retry with exponential backoff
  - Rate limit detection (HTTP 429/503) → auto long pause
  - Parallel workers (configurable, default 3)
  - Failed-video log for easy re-run
  - Adaptive inter-download delay that grows when errors spike
"""
import os
import subprocess
import sys
import time
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Tunable constants ────────────────────────────────────────────────────────
NUM_WORKERS          = 3      # parallel download threads (keep ≤ 3 to stay safe)
BASE_DELAY_SEC       = 3      # base sleep between downloads (per worker)
MAX_DELAY_SEC        = 30     # max sleep after back-off
MAX_RETRIES          = 3      # per-video retry attempts
RETRY_BACKOFF_BASE   = 10     # first retry waits this many seconds
RATE_LIMIT_PAUSE_SEC = 180    # global pause when 429/rate-limit detected
# ─────────────────────────────────────────────────────────────────────────────

# Global flag + lock for rate-limit coordination across threads
_rate_limit_event = threading.Event()
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


def read_video_ids(file_path):
    """Read video IDs from file."""
    video_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            video_id = line.strip()
            if video_id:
                video_ids.append(video_id)
    return video_ids


def is_valid_video_file(file_path, require_mp4=False):
    """Check if a file is a valid video file (MP4, TS, MKV, WebM)."""
    if not file_path.exists():
        return False
    if file_path.stat().st_size == 0:
        return False
    try:
        with open(file_path, 'rb') as f:
            header = f.read(20)
            if require_mp4:
                return b'ftyp' in header
            else:
                return (
                    b'ftyp' in header or
                    header[0] == 0x47 or
                    header[:4] == b'\x1a\x45\xdf\xa3'
                )
    except Exception:
        return False


def find_downloaded_file(video_id, output_dir, check_valid=True, require_mp4=False):
    """Find downloaded video file with any extension."""
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
    """Remove invalid or corrupted video files."""
    cleaned = []
    for ext in ['.mp4', '.mkv', '.webm', '.ts', '.part', '.ytdl']:
        check_file = output_dir / f"{video_id}{ext}"
        if check_file.exists():
            if ext in ['.mp4', '.mkv', '.webm', '.ts']:
                if not is_valid_video_file(check_file):
                    try:
                        check_file.unlink()
                        cleaned.append(str(check_file))
                    except Exception as e:
                        tprint(f"[WARNING] Could not delete invalid file {check_file}: {e}")
            else:
                try:
                    check_file.unlink()
                    cleaned.append(str(check_file))
                except Exception as e:
                    tprint(f"[WARNING] Could not delete temp file {check_file}: {e}")
    return cleaned


def is_rate_limited(stderr_text: str) -> bool:
    """Detect real rate-limit / bot-detection signals in yt-dlp stderr.
    Keep this list narrow to avoid false positives.
    """
    keywords = [
        'HTTP Error 429',
        'Too Many Requests',
        'Sign in to confirm your age',
        'Precondition Failed',
    ]
    lower = stderr_text.lower()
    return any(kw.lower() in lower for kw in keywords)


def is_unavailable(stderr_text: str) -> bool:
    """Detect permanently unavailable videos (no point retrying)."""
    keywords = [
        'Video unavailable',
        'Private video',
        'has been removed',
        'This video is not available',
        'account associated with',
        'copyright',
        'age-restricted',
    ]
    lower = stderr_text.lower()
    return any(kw.lower() in lower for kw in keywords)


def build_cmd(python_exe, youtube_dl_path, url, output_template, cookies_file):
    """Build yt-dlp command with anti-rate-limit options."""
    base = python_exe and [python_exe, '-m', 'yt_dlp'] or [youtube_dl_path]

    cmd = base + [
        url,
        # Explicitly exclude HLS (m3u8) formats — YouTube HLS segments require
        # a GVS PO Token; without it fragments come back empty.
        # Prefer DASH mp4 (avc1 video + m4a audio), fall back to any non-HLS mp4.
        '-f', (
            'bestvideo[ext=mp4][vcodec^=avc1][protocol!=m3u8][protocol!=m3u8_native]'
            '+bestaudio[ext=m4a][protocol!=m3u8][protocol!=m3u8_native]'
            '/bestvideo[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]'
            '+bestaudio[ext=m4a][protocol!=m3u8][protocol!=m3u8_native]'
            '/best[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]'
            '/best[protocol!=m3u8][protocol!=m3u8_native]'
        ),
        '--merge-output-format', 'mp4',
        '--no-check-certificate',
        '--restrict-filenames',
        '--prefer-free-formats',
        '--recode-video', 'mp4',
        '-o', output_template,

        # ── Anti-rate-limit / robustness options ──────────────────────────
        '--retries',           '5',
        '--fragment-retries',  '5',
        '--extractor-retries', '3',
        '--sleep-interval',    '3',
        '--max-sleep-interval','8',
        '--no-abort-on-error',
        '--js-runtimes', 'node',
    ]

    if cookies_file and Path(cookies_file).exists():
        cmd.extend(['--cookies', str(cookies_file)])

    return cmd


def download_video(video_id, output_dir, python_exe, youtube_dl_path, cookies_file, env):
    """
    Download one video with retry + exponential backoff.
    Returns: 'ok' | 'skip' | 'fail' | 'unavailable'
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(output_dir / "%(id)s.%(ext)s")

    # ── Clean up leftovers ───────────────────────────────────────────────────
    cleaned = cleanup_invalid_files(video_id, output_dir)
    if cleaned:
        tprint(f"  [CLEANUP] Removed: {', '.join(Path(f).name for f in cleaned)}")

    # ── Already exists? ──────────────────────────────────────────────────────
    existing = find_downloaded_file(video_id, output_dir, require_mp4=True)
    if not existing:
        existing = find_downloaded_file(video_id, output_dir, require_mp4=False)
        if existing and existing.suffix != '.mp4':
            tprint(f"  [RE-DL] Found {existing.suffix.upper()}, re-downloading as MP4")
            try:
                existing.unlink()
            except Exception:
                pass
            existing = None

    if existing:
        sz = existing.stat().st_size
        fmt = existing.suffix[1:].upper()
        tprint(f"  [SKIP] Already valid ({fmt}, {sz/1024/1024:.2f} MB)")
        return 'skip'

    cmd = build_cmd(python_exe, youtube_dl_path, url, output_template, cookies_file)

    for attempt in range(1, MAX_RETRIES + 1):
        # Honour global rate-limit pause
        if _rate_limit_event.is_set():
            tprint(f"  [WAIT] Rate-limit pause active, sleeping {RATE_LIMIT_PAUSE_SEC}s …")
            time.sleep(RATE_LIMIT_PAUSE_SEC)
            _rate_limit_event.clear()

        try:
            result = subprocess.run(
                cmd,
                shell=False,
                timeout=600,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            tprint(f"  [TIMEOUT] attempt {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 5)
                tprint(f"  [RETRY] waiting {wait:.1f}s …")
                time.sleep(wait)
            continue
        except Exception as e:
            tprint(f"  [ERROR] {e}")
            return 'fail'

        stderr = result.stderr or ""
        stdout = result.stdout or ""
        combined = stderr + stdout

        # Permanently unavailable — no retry
        if is_unavailable(combined):
            tprint(f"  [UNAVAILABLE] {video_id}: video is private/removed/restricted")
            return 'unavailable'

        # Rate limited — trigger global pause + retry
        if result.returncode != 0 and is_rate_limited(combined):
            tprint(f"  [RATE-LIMIT] detected on {video_id}, triggering global pause …")
            _rate_limit_event.set()
            time.sleep(RATE_LIMIT_PAUSE_SEC + random.uniform(0, 15))
            _rate_limit_event.clear()
            if attempt < MAX_RETRIES:
                continue
            return 'fail'

        if result.returncode != 0:
            err_snippet = stderr[:400] if stderr else "Unknown error"
            tprint(f"  [ERROR] attempt {attempt}/{MAX_RETRIES}: {err_snippet}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 5)
                tprint(f"  [RETRY] waiting {wait:.1f}s …")
                time.sleep(wait)
            continue

        # Download succeeded — verify file
        time.sleep(2)
        dl = find_downloaded_file(video_id, output_dir, check_valid=True, require_mp4=True)
        if not dl:
            dl = find_downloaded_file(video_id, output_dir, check_valid=True, require_mp4=False)

        if dl:
            sz = dl.stat().st_size
            fmt = dl.suffix[1:].upper()
            if fmt != 'MP4':
                tprint(f"  [WARNING] Got {fmt} instead of MP4: {video_id} ({sz/1024/1024:.2f} MB)")
            else:
                tprint(f"  [OK] {video_id} ({sz/1024/1024:.2f} MB, {fmt})")
            return 'ok'
        else:
            tprint(f"  [ERROR] File missing after apparent success: {video_id}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE)
            continue

    return 'fail'


def worker_task(args):
    """Wrapper for ThreadPoolExecutor: adds per-worker base delay."""
    idx, total, video_id, output_dir, python_exe, youtube_dl_path, cookies_file, env = args
    tprint(f"\n[{idx}/{total}] {video_id}")
    status = download_video(video_id, output_dir, python_exe, youtube_dl_path, cookies_file, env)
    # Randomised delay to spread worker requests (avoids burst pattern)
    delay = BASE_DELAY_SEC + random.uniform(0, MAX_DELAY_SEC - BASE_DELAY_SEC)
    time.sleep(delay)
    return video_id, status


def main():
    # ── Resolve yt-dlp ────────────────────────────────────────────────────────
    try:
        subprocess.run([sys.executable, '-m', 'yt_dlp', '--version'],
                       capture_output=True, check=True, timeout=5)
        python_exe      = sys.executable
        youtube_dl_path = None
        print(f"Using: {python_exe} -m yt_dlp")
    except Exception:
        python_exe      = None
        youtube_dl_path = 'yt-dlp'
        print("Using: yt-dlp")

    # ── Input file ────────────────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    input_file = os.getenv('YOUTUBE_DOWNLOAD_INPUT_FILE')
    video_ids_file = Path(input_file) if input_file else script_dir / "youtube_video_ids_stage2_notstrict.txt"

    if not video_ids_file.exists():
        print(f"[ERROR] Video IDs file not found: {video_ids_file}")
        return

    print(f"Reading video IDs from: {video_ids_file}")
    video_ids = read_video_ids(video_ids_file)
    print(f"Found {len(video_ids)} video IDs")

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir_str = os.getenv('YOUTUBE_DOWNLOAD_OUTPUT_DIR')
    output_dir = Path(output_dir_str) if output_dir_str else Path("/shared/rc/llm-gen-agent/mhu/videos/youtube/sources")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Cookies ───────────────────────────────────────────────────────────────
    cookies_file = script_dir / "www.youtube.com_cookies.txt"
    if not cookies_file.exists():
        cookies_file = script_dir.parent / "www.youtube.com_cookies.txt"
    if cookies_file.exists():
        print(f"Using cookies: {cookies_file}")
    else:
        print("No cookies file found (proceeding without cookies)")
        cookies_file = None

    # ── Failure log ───────────────────────────────────────────────────────────
    failed_log   = script_dir / "failed_video_ids.txt"
    unavail_log  = script_dir / "unavailable_video_ids.txt"

    # ── Env for subprocess ────────────────────────────────────────────────────
    env = os.environ.copy()
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        lib_path = os.path.join(conda_prefix, 'lib')
        cur = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f"{lib_path}:{cur}" if cur else lib_path

    print(f"\n{'='*60}")
    print(f"Starting downloads | workers={NUM_WORKERS} | base_delay={BASE_DELAY_SEC}s")
    print(f"{'='*60}\n")

    success_count   = 0
    fail_count      = 0
    skip_count      = 0
    unavail_count   = 0
    failed_ids      = []
    unavail_ids     = []

    total = len(video_ids)
    tasks = [
        (i, total, vid, output_dir, python_exe, youtube_dl_path, cookies_file, env)
        for i, vid in enumerate(video_ids, 1)
    ]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(worker_task, t): t for t in tasks}
        for future in as_completed(futures):
            video_id, status = future.result()
            if status == 'ok':
                success_count += 1
            elif status == 'skip':
                skip_count += 1
            elif status == 'unavailable':
                unavail_count += 1
                unavail_ids.append(video_id)
            else:
                fail_count += 1
                failed_ids.append(video_id)

            done = success_count + fail_count + skip_count + unavail_count
            if done % 50 == 0:
                tprint(f"\n── Progress {done}/{total} | ✓{success_count} skip{skip_count} ✗{fail_count} unavail{unavail_count} ──\n")

    # ── Save failure / unavailable logs ──────────────────────────────────────
    if failed_ids:
        with open(failed_log, 'w') as f:
            f.write('\n'.join(failed_ids) + '\n')
        print(f"\n[LOG] Failed IDs saved to: {failed_log}  ({len(failed_ids)} videos)")
        print(f"      Re-run with: YOUTUBE_DOWNLOAD_INPUT_FILE={failed_log}")

    if unavail_ids:
        with open(unavail_log, 'w') as f:
            f.write('\n'.join(unavail_ids) + '\n')
        print(f"[LOG] Unavailable IDs saved to: {unavail_log}  ({len(unavail_ids)} videos)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"  Total:       {total}")
    print(f"  Success:     {success_count}")
    print(f"  Skipped:     {skip_count}")
    print(f"  Failed:      {fail_count}")
    print(f"  Unavailable: {unavail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
