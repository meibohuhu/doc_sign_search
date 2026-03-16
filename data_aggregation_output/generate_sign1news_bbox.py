"""
为 Sign1News video clips 生成 bbox 文件
参考 clips_bbox.py，使用 YOLO 检测人，光流法跟踪，生成归一化的 bbox
输出格式与 bbox-v1.0.json 一致


"""

import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO


def get_optical_flow(images):
    """计算光流，用于跟踪目标"""
    prv_gray = None
    motion_mags = []
    for frame in images:
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_size = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))
        cur_gray = cv2.resize(cur_gray, gray_size)
        if prv_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prv_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = (255.0*(mag-mag.min())/max(float(mag.max()-mag.min()), 1)).astype(np.uint8)
            mag = cv2.resize(mag, (frame.shape[1], frame.shape[0]))
        else:
            mag = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        prv_gray = cur_gray
        motion_mags.append(mag)
    return motion_mags


def get_iou(boxA, boxB):
    """计算两个 bbox 的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def find_target_bbox(bbox_arr, opts, iou_thr=0.5, len_ratio_thr=0.5):
    """
    使用光流法找到目标 bbox（通常是手语者）
    返回平均 bbox 和所有 tubes
    """
    tubes = []
    num_rest = sum([len(x) for x in bbox_arr])
    while num_rest > 0:
        for i, bboxes in enumerate(bbox_arr):
            if len(bboxes) > 0:
                anchor = [i, bbox_arr[i].pop()]
                break
        tube = [anchor]
        for i in range(len(bbox_arr)):
            bboxes = bbox_arr[i]
            if anchor[0] == i or len(bboxes) == 0:
                continue
            ious = np.array([get_iou(anchor[1], bbox) for bbox in bboxes])
            j = ious.argmax()
            if ious[j] > iou_thr:
                target_bbox = bboxes.pop(j)
                tube.append([i, target_bbox])
        tubes.append(tube)
        num_rest = sum([len(x) for x in bbox_arr])
        
    max_val, max_tube = -1, None
    for itube, tube in enumerate(tubes):
        mean_val = 0
        for iframe, bbox in tube:
            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            mean_val += opts[iframe][max(y0, 0): y1, max(x0, 0): x1].mean()
        mean_val /= len(tube)
        
        if len(tube)/len(opts) > len_ratio_thr:
            if mean_val > max_val:
                max_val, max_tube = mean_val, tube
                
    if max_tube is not None:
        target_bbox = np.array([bbox[1] for bbox in max_tube]).mean(axis=0).tolist()
    else:
        target_bbox = None
    return target_bbox, tubes


def adjust_bbox_for_upper_body(bbox, width, height):
    """
    调整 bbox 以包含上半身（特别是手和脸）
    对于手语视频，通常需要：
    - 向上扩展以包含头部
    - 向左右扩展以包含手部动作
    - 向下稍微扩展（主要是上半身）
    """
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    
    # 扩展参数（相对于 bbox 大小）
    # 大幅增加扩展参数，使 bbox 更大以包含更多背景和上下文
    # 左边留少一些，右边留多一些，上面多留一些
    up_exp = 0.8    # 向上扩展 80% 以包含头部和更多上方背景
    down_exp = 0.3  # 向下扩展 30%（包含更多上半身和背景）
    left_exp = 0.25  # 向左扩展 25%（减少左边扩展）
    right_exp = 0.55 # 向右扩展 55%（增加右边扩展，包含更多手部动作和背景）
    
    # 计算新的 bbox
    new_x0 = x0 - w * left_exp
    new_y0 = y0 - h * up_exp
    new_x1 = x1 + w * right_exp
    new_y1 = y1 + h * down_exp
    
    # 添加边界边距（留 1% 的边距，避免 bbox 正好到达边界）
    margin_x = width * 0.01
    margin_y = height * 0.01
    
    # 限制在边界内，但留出边距
    new_x0 = max(margin_x, new_x0)
    new_y0 = max(margin_y, new_y0)
    new_x1 = min(width - margin_x, new_x1)
    new_y1 = min(height - margin_y, new_y1)
    
    # 确保 bbox 合理（不要覆盖整个图像）
    # 如果 bbox 太大（超过图像的 90%），则限制它
    bbox_width = new_x1 - new_x0
    bbox_height = new_y1 - new_y0
    max_width = width * 0.90  # 允许更大的 bbox，包含更多背景
    max_height = height * 0.90
    
    if bbox_width > max_width or bbox_height > max_height:
        # 如果太大，使用更保守的扩展，以中心为基准
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        new_w = min(bbox_width, max_width)
        new_h = min(bbox_height, max_height)
        new_x0 = max(margin_x, center_x - new_w / 2)
        new_y0 = max(margin_y, center_y - new_h / 2)
        new_x1 = min(width - margin_x, new_x0 + new_w)
        new_y1 = min(height - margin_y, new_y0 + new_h)
    
    # 最终检查：确保 bbox 有效且合理
    if new_x1 <= new_x0 or new_y1 <= new_y0:
        # 如果无效，返回原始 bbox（稍微扩展）
        return [max(0, x0 - w * 0.1), max(0, y0 - h * 0.1), 
                min(width, x1 + w * 0.1), min(height, y1 + h * 0.1)]
    
    return [new_x0, new_y0, new_x1, new_y1]


def get_bbox_for_clip(video_path, yolo_model_path, _model_cache={}):
    """
    为单个视频 clip 生成 bbox。
    使用 YOLO-Pose 检测每个人的手腕关键点运动量，选出做手语的人。
    单人时直接取平均 bbox；多人时选手腕运动量最大的人。
    返回归一化的 bbox [x0, y0, x1, y1]（相对于图像宽高）
    """
    # COCO pose 关键点索引：9=左手腕, 10=右手腕
    WRIST_KP = [9, 10]

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 加载 YOLO-Pose 模型（每个进程只加载一次）
        pose_model_path = yolo_model_path.replace("yolov8n.pt", "yolov8n-pose.pt")
        if "pose" not in pose_model_path:
            pose_model_path = "yolov8n-pose.pt"
        if pose_model_path not in _model_cache:
            _model_cache[pose_model_path] = YOLO(pose_model_path)
        model = _model_cache[pose_model_path]

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return None

        # 批量推理
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        batch_results = model(frames_rgb, verbose=False)

        # 收集每帧每个人的 bbox 和手腕坐标
        # per_person[person_idx] = {'bboxes': [...], 'wrists': [...]}
        # 用跨帧 track_id 对应同一个人；无 track 时按帧内顺序对应
        # 简化：取每帧所有人，最后按手腕运动量排名
        all_frame_data = []  # list of list of (bbox, wrist_pts)
        for results in batch_results:
            frame_persons = []
            if results.keypoints is not None and len(results.boxes) > 0:
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    kps = results.keypoints.xy[i]  # shape [17, 2]
                    wrists = []
                    for kp_idx in WRIST_KP:
                        kx, ky = kps[kp_idx].tolist()
                        if kx > 0 and ky > 0:  # 0,0 表示未检测到
                            wrists.append((kx, ky))
                    frame_persons.append(([x1, y1, x2, y2], wrists))
            all_frame_data.append(frame_persons)

        # 只有一个人时直接取平均
        max_persons = max((len(fp) for fp in all_frame_data), default=0)
        if max_persons == 0:
            return None

        if max_persons == 1:
            single_bboxes = [fp[0][0] for fp in all_frame_data if len(fp) == 1]
            bbox = np.array(single_bboxes).mean(axis=0).tolist()
            tubes = []
        else:
            # 多人：按手腕运动量选手语者
            # 策略：对每帧每个人累计手腕位移，选运动量最大的人
            # 用简单的帧内 index 对应（同一位置的人视为同一人）
            # 先统计每个"slot"的手腕运动总量
            num_slots = max_persons
            slot_wrist_motion = [0.0] * num_slots  # 每个 slot 的手腕运动总量
            slot_bboxes = [[] for _ in range(num_slots)]
            prev_wrists = [None] * num_slots

            for frame_persons in all_frame_data:
                # 按 bbox 面积降序排列，保证 slot 对应稳定
                frame_persons_sorted = sorted(
                    frame_persons,
                    key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]),
                    reverse=True
                )
                for slot_idx, (bbox_p, wrists) in enumerate(frame_persons_sorted[:num_slots]):
                    slot_bboxes[slot_idx].append(bbox_p)
                    if wrists and prev_wrists[slot_idx] is not None:
                        for (cx, cy), (px, py) in zip(wrists, prev_wrists[slot_idx]):
                            slot_wrist_motion[slot_idx] += np.sqrt((cx-px)**2 + (cy-py)**2)
                    prev_wrists[slot_idx] = wrists if wrists else prev_wrists[slot_idx]

            # 选手腕运动量最大的 slot
            best_slot = int(np.argmax(slot_wrist_motion))
            if not slot_bboxes[best_slot]:
                # fallback：选面积最大的
                best_slot = 0
            bbox = np.array(slot_bboxes[best_slot]).mean(axis=0).tolist()
            tubes = []
        
        # 如果没找到，尝试从 tubes 中选择面积最大的
        if bbox is None and len(tubes) > 0:
            valid_tubes = [tube for tube in tubes if len(tube) > 0]
            if len(valid_tubes) > 0:
                # 选择平均面积最大的 tube
                max_avg_area = -1
                best_tube = None
                for tube in valid_tubes:
                    avg_area = np.mean([get_bbox_area(bbox[1]) for bbox in tube])
                    if avg_area > max_avg_area:
                        max_avg_area = avg_area
                        best_tube = tube
                if best_tube is not None:
                    bbox = np.array([x for _, x in best_tube]).mean(axis=0).tolist()
        
        if bbox is None:
            return None
        
        # 调整 bbox 以包含上半身（手和脸）
        # 这是主要的扩展步骤，针对手语视频优化
        bbox = adjust_bbox_for_upper_body(bbox, width, height)
        
        # 归一化到 [0, 1] 范围（相对于图像宽高）
        normalized_bbox = [
            bbox[0] / width,   # x0
            bbox[1] / height,  # y0
            bbox[2] / width,   # x1
            bbox[3] / height   # y1
        ]
        
        return normalized_bbox
        
    except Exception as e:
        print(f"错误处理 {video_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="为 Sign1News video clips 生成 bbox 文件")
    parser.add_argument(
        "--clips-dir",
        type=str,
        default=None,
        help="视频 clips 目录（如果提供，会处理目录中的所有 .mp4 文件）"
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default=None,
        help="单个视频文件路径（如果提供，只处理这个文件）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sign1news/sign1news_bbox.json",
        help="输出 bbox JSON 文件路径"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO 模型路径（默认使用 yolov8n.pt）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行处理的工作进程数（默认 1）"
    )
    
    args = parser.parse_args()
    
    # 确定要处理的视频文件
    if args.video_file:
        # 处理单个文件
        video_file_path = Path(args.video_file)
        if not video_file_path.exists():
            print(f"错误: 视频文件不存在: {args.video_file}")
            return
        if not video_file_path.suffix.lower() == '.mp4':
            print(f"错误: 文件不是 .mp4 格式: {args.video_file}")
            return
        video_files = [video_file_path]
    elif args.clips_dir:
        # 处理目录中的所有视频文件
        clips_dir = Path(args.clips_dir)
        video_files = list(clips_dir.glob("*.mp4"))
        if len(video_files) == 0:
            print(f"错误: 在 {args.clips_dir} 中未找到视频文件")
            return
    else:
        print("错误: 必须提供 --clips-dir 或 --video-file 参数")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"YOLO 模型: {args.yolo_model}")
    print(f"输出文件: {args.output}")
    print("-" * 50)
    
    # 生成 bbox
    bbox_dict = {}
    failed_clips = []

    def _process(video_file):
        clip_id = video_file.stem
        bbox = get_bbox_for_clip(str(video_file), args.yolo_model)
        return clip_id, bbox

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process, vf): vf for vf in video_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理视频"):
            clip_id, bbox = future.result()
            if bbox is not None:
                bbox_dict[clip_id] = bbox
            else:
                failed_clips.append(clip_id)
                print(f"  ⚠ 无法生成 bbox: {clip_id}")
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_dict, f, indent=4, ensure_ascii=False)
    
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"成功: {len(bbox_dict)} 个 clips")
    print(f"失败: {len(failed_clips)} 个 clips")
    print(f"输出文件: {args.output}")
    
    if len(failed_clips) > 0:
        print(f"\n失败的 clips:")
        for clip_id in failed_clips[:10]:  # 只显示前10个
            print(f"  - {clip_id}")
        if len(failed_clips) > 10:
            print(f"  ... 还有 {len(failed_clips) - 10} 个")


if __name__ == "__main__":
    main()

