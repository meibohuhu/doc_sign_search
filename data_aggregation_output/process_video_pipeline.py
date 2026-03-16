#!/usr/bin/env python3
"""
视频处理流水线脚本（多 worker 并行版）
按照以下顺序处理每个视频：
1. clip_sign1news_videos.py  - 根据 caption timestamps 裁剪视频
2. generate_sign1news_bbox.py - 生成 bbox
3. visualize_bbox.py          - 使用 bbox 裁剪视频到 224x224

并行模式：每个 worker 独立处理一个原始视频（step1→step2→step3），最后合并 bbox。 
每个 worker 跑一个完整的 per-video pipeline。我来修改：
output/
├── clips/              # 所有视频的 clips（共享）
├── bbox/               # 每个视频单独的 bbox json（video_id.json）
├── clips_bbox.json     # 最终合并的总 bbox（pipeline 结束时生成）
└── clips_cropped_224/  # 所有裁剪后的视频

用法示例:
  # 第一次运行（4 个 worker）
  python process_video_pipeline.py \
      --input-dir /scratch/mh2803/source/youtube_asl_clips_video_ids_2 \
      --output-dir /scratch/mh2803/source/youtube_asl_clips_processed_2 \
      --metadata youtube-asl_metadata.csv \
      --workers 4

  # 中途失败后续跑（自动跳过已完成的视频）
  python process_video_pipeline.py ... --workers 4 --resume

  # 指定日志文件
  python process_video_pipeline.py ... --log-file /scratch/mh2803/pipeline.log
"""

import os
import sys
import json
import subprocess
import argparse
import logging
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_print_lock = threading.Lock()


def setup_logger(log_file=None):
    """设置日志：同时输出到终端和文件"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    return logging.getLogger(__name__)


def tlog(logger, msg):
    """线程安全日志"""
    with _print_lock:
        logger.info(msg)


def run_command(cmd, description, video_id):
    """运行子命令，返回 (success, stderr)"""
    result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr[:500]
    return True, ""


def process_one_video(video_file, output_base, metadata_path, script_dir,
                      python_exe, target_size, resume, logger):
    """
    对单个原始视频跑完整 3 步 pipeline。
    输出结构:
      <output_base>/clips/<video_id>_*.mp4
      <output_base>/bbox/<video_id>.json
      <output_base>/clips_cropped_224/<video_id>_*.mp4
    返回: (video_id, 'ok' | 'skip' | 'fail')
    """
    video_id   = video_file.stem
    clips_dir  = output_base / "clips"
    bbox_dir   = output_base / "bbox"
    bbox_file  = bbox_dir / f"{video_id}.json"
    cropped_dir = output_base / "clips_cropped_224"

    clips_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume 检查：这个视频是否已全部完成 ─────────────────────────────────
    if resume:
        existing_clips   = list(clips_dir.glob(f"{video_id}_*.mp4"))
        existing_cropped = list(cropped_dir.glob(f"{video_id}_*.mp4"))
        if bbox_file.exists() and existing_clips and existing_cropped:
            tlog(logger, f"  [skip] {video_id} 已完成，跳过")
            return video_id, "skip"

    # ── Step 1: Clip ─────────────────────────────────────────────────────────
    cmd1 = [
        python_exe, str(script_dir / "clip_sign1news_videos.py"),
        "--metadata", str(metadata_path),
        "--input", str(video_file),
        "--output-dir", str(clips_dir),
    ]
    ok, err = run_command(cmd1, "clip", video_id)
    if not ok:
        tlog(logger, f"  [fail] {video_id} step1 clip 失败: {err}")
        return video_id, "fail"

    new_clips = list(clips_dir.glob(f"{video_id}_*.mp4"))
    if not new_clips:
        tlog(logger, f"  [warn] {video_id} step1 未生成任何 clip，跳过")
        return video_id, "skip"

    # ── Step 2: BBox ──────────────────────────────────────────────────────────
    # 对这个视频的 clips 单独生成 bbox
    # 用临时目录只放这个视频的 clips，避免与其他 worker 冲突
    tmp_clips_dir = output_base / "tmp_clips" / video_id
    tmp_clips_dir.mkdir(parents=True, exist_ok=True)
    for c in new_clips:
        dst = tmp_clips_dir / c.name
        if not dst.exists():
            dst.symlink_to(c.resolve())

    cmd2 = [
        python_exe, str(script_dir / "generate_sign1news_bbox.py"),
        "--clips-dir", str(tmp_clips_dir),
        "--output", str(bbox_file),
    ]
    ok, err = run_command(cmd2, "bbox", video_id)
    if not ok:
        tlog(logger, f"  [fail] {video_id} step2 bbox 失败: {err}")
        return video_id, "fail"

    # ── Step 3: Crop ──────────────────────────────────────────────────────────
    cmd3 = [
        python_exe, str(script_dir / "visualize_bbox.py"),
        "--bbox-file", str(bbox_file),
        "--clips-dir", str(clips_dir),
        "--output-dir", str(cropped_dir),
        "--target-size", str(target_size),
    ]
    ok, err = run_command(cmd3, "crop", video_id)
    if not ok:
        tlog(logger, f"  [fail] {video_id} step3 crop 失败: {err}")
        return video_id, "fail"

    cropped = list(cropped_dir.glob(f"{video_id}_*.mp4"))
    tlog(logger, f"  [ok] {video_id}: {len(new_clips)} clips → {len(cropped)} cropped")
    return video_id, "ok"


def merge_bbox(output_base, logger):
    """合并所有 per-video bbox json 到 clips_bbox.json"""
    bbox_dir = output_base / "bbox"
    merged = {}
    for f in sorted(bbox_dir.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            merged.update(data)
        except Exception as e:
            logger.info(f"  [warn] 无法读取 bbox 文件 {f.name}: {e}")

    out_file = output_base / "clips_bbox.json"
    with open(out_file, "w") as fh:
        json.dump(merged, fh, indent=2)
    logger.info(f"✅ 合并 bbox: {len(merged)} 条记录 → {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="视频处理流水线（多 worker 并行版）")
    parser.add_argument("--input-dir",   type=str, required=True, help="输入视频目录")
    parser.add_argument("--output-dir",  type=str, required=True, help="输出目录")
    parser.add_argument("--metadata",    type=str, default="youtube-asl_metadata.csv",
                        help="metadata CSV 文件路径")
    parser.add_argument("--target-size", type=int, default=224, help="输出视频尺寸（默认 224）")
    parser.add_argument("--conda-env",   type=str, default="internvl",
                        help="Conda 环境名称（用于自动推断 python 路径）")
    parser.add_argument("--python",      type=str, default=None,
                        help="Python 可执行文件路径（默认自动从 conda-env 推断）")
    parser.add_argument("--workers",     type=int, default=4,
                        help="并行 worker 数量（默认 4，建议 CPU 核数的 1/2）")
    parser.add_argument("--log-file",    type=str, default=None,
                        help="日志文件路径（默认: <output-dir>/pipeline_<timestamp>.log）")
    parser.add_argument("--resume",      action="store_true",
                        help="跳过已完成的视频，从中断处继续")
    parser.add_argument("--skip-clip",   action="store_true", help="跳过所有视频的步骤1")
    parser.add_argument("--skip-bbox",   action="store_true", help="跳过所有视频的步骤2")
    parser.add_argument("--skip-crop",   action="store_true", help="跳过所有视频的步骤3")

    args = parser.parse_args()

    # ── 输出 & 日志 ───────────────────────────────────────────────────────────
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    log_path = args.log_file or str(
        output_base / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = setup_logger(log_path)

    # ── 检查输入 ──────────────────────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.info(f"❌ 输入目录不存在: {input_dir}")
        return 1

    script_dir   = Path(__file__).parent
    project_root = script_dir.parent
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = project_root / metadata_path
    if not metadata_path.exists():
        logger.info(f"❌ metadata 文件不存在: {metadata_path}")
        return 1

    video_files = sorted(input_dir.glob("*.mp4"))
    if not video_files:
        logger.info(f"❌ 在 {input_dir} 中没有找到 .mp4 文件")
        return 1

    # ── 解析 python 路径 ───────────────────────────────────────────────────────
    if args.python:
        python_exe = args.python
    else:
        home = os.environ.get("HOME", "/home/stu2/s15/mh2803")
        python_exe = f"{home}/anaconda3/envs/{args.conda_env}/bin/python3.10"
    if not Path(python_exe).exists():
        logger.info(f"❌ Python 可执行文件不存在: {python_exe}")
        logger.info(f"   请使用 --python 指定正确路径")
        return 1

    # ── 打印配置 ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("视频处理流水线（多 worker 并行版）")
    logger.info(f"{'='*60}")
    logger.info(f"输入目录:  {input_dir}")
    logger.info(f"输出目录:  {output_base}")
    logger.info(f"Metadata:  {metadata_path}")
    logger.info(f"Python:    {python_exe}")
    logger.info(f"目标尺寸:  {args.target_size}x{args.target_size}")
    logger.info(f"Workers:   {args.workers}")
    logger.info(f"总视频数:  {len(video_files)}")
    logger.info(f"Resume:    {args.resume}")
    logger.info(f"日志文件:  {log_path}")
    logger.info(f"{'='*60}\n")

    # ── 并行处理 ──────────────────────────────────────────────────────────────
    ok_count   = 0
    skip_count = 0
    fail_count = 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_video,
                vf, output_base, metadata_path, script_dir,
                python_exe, args.target_size, args.resume, logger
            ): vf for vf in video_files
        }

        total = len(futures)
        done  = 0
        for future in as_completed(futures):
            video_id, status = future.result()
            done += 1
            if status == "ok":
                ok_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1
                failed_ids.append(video_id)

            if done % 50 == 0 or done == total:
                logger.info(
                    f"\n── 进度 {done}/{total} | "
                    f"✓{ok_count} skip{skip_count} ✗{fail_count} ──\n"
                )

    # ── 合并 bbox ─────────────────────────────────────────────────────────────
    logger.info("\n合并所有 bbox 文件...")
    merge_bbox(output_base, logger)

    # ── 失败记录 ──────────────────────────────────────────────────────────────
    if failed_ids:
        fail_log = output_base / "failed_video_ids.txt"
        fail_log.write_text("\n".join(failed_ids) + "\n")
        logger.info(f"\n[LOG] 失败视频 ID 已保存到: {fail_log}")

    # ── 总结 ──────────────────────────────────────────────────────────────────
    clips_count   = len(list((output_base / "clips").glob("*.mp4"))) if (output_base / "clips").exists() else 0
    cropped_count = len(list((output_base / "clips_cropped_224").glob("*.mp4"))) if (output_base / "clips_cropped_224").exists() else 0

    logger.info(f"\n{'='*60}")
    logger.info("✅ 处理完成！")
    logger.info(f"{'='*60}")
    logger.info(f"  原始视频:     {total}")
    logger.info(f"  成功:         {ok_count}")
    logger.info(f"  跳过(已完成): {skip_count}")
    logger.info(f"  失败:         {fail_count}")
    logger.info(f"  生成 clips:   {clips_count}")
    logger.info(f"  Cropped 224:  {cropped_count}")
    logger.info(f"输出目录结构:")
    logger.info(f"  clips/              → {output_base / 'clips'}")
    logger.info(f"  bbox/               → {output_base / 'bbox'}")
    logger.info(f"  clips_bbox.json     → {output_base / 'clips_bbox.json'}")
    logger.info(f"  clips_cropped_224/  → {output_base / 'clips_cropped_224'}")
    logger.info(f"  pipeline log:       → {log_path}")
    logger.info(f"{'='*60}\n")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
