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


def get_bbox_for_clip(video_path, yolo_model_path):
    """
    为单个视频 clip 生成 bbox
    返回归一化的 bbox [x0, y0, x1, y1]（相对于图像宽高）
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 加载 YOLO 模型
        model = YOLO(yolo_model_path)
        
        bboxes = []
        frames = []
        
        # 读取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为 RGB（YOLO 需要 RGB）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 运行 YOLO 推理
            results = model(frame_rgb)
            
            # 过滤出 person 类别（COCO 数据集中 class 0）
            person_bboxes = []
            for r in results:
                for box in r.boxes:
                    if box.cls == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        person_bboxes.append([x1, y1, x2, y2])
            
            bboxes.append(person_bboxes)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # 轻微扩展 bbox 以包含更多上下文（用于跟踪）
        # 注意：这里只做轻微扩展，主要扩展在后面的 adjust_bbox_for_upper_body 中
        # 左边留少一些，右边留多一些
        up_exp, down_exp, left_exp, right_exp = 0.1, 0.1, 0.1, 0.2
        # 添加小的边界边距
        margin_x = width * 0.01
        margin_y = height * 0.01
        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                x0, y0, x1, y1 = bboxes[i][j]
                w, h = x1 - x0 + 1, y1 - y0 + 1
                x0, y0, x1, y1 = x0 - w * left_exp, y0 - h * up_exp, x1 + w * right_exp, y1 + h * down_exp
                # 确保不超出边界，但留出边距
                x0 = max(margin_x, x0)
                y0 = max(margin_y, y0)
                x1 = min(width - margin_x, x1)
                y1 = min(height - margin_y, y1)
                bboxes[i][j] = [x0, y0, x1, y1]
        
        # 找到目标 bbox - 优先选择最大的人（通常是主要手语者）
        # 计算每个人的平均面积
        def get_bbox_area(bbox):
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # 如果每帧都只有一个 person，直接取平均
        if max([len(x) for x in bboxes]) == 1:
            bboxes = list(filter(lambda x: len(x) == 1, bboxes))
            if len(bboxes) == 0:
                return None
            bbox = np.array(bboxes).mean(axis=0)[0].tolist()
            tubes = []
        else:
            # 有多个人时，选择面积最大的人
            # 首先，计算每帧中最大的人
            max_person_bboxes = []
            for frame_bboxes in bboxes:
                if len(frame_bboxes) == 0:
                    continue
                # 选择面积最大的人
                areas = [get_bbox_area(b) for b in frame_bboxes]
                max_idx = np.argmax(areas)
                max_person_bboxes.append(frame_bboxes[max_idx])
            
            if len(max_person_bboxes) > 0:
                # 取所有帧中最大的人的平均 bbox
                bbox = np.array(max_person_bboxes).mean(axis=0).tolist()
                tubes = []
            else:
                # 如果还是没找到，使用光流法跟踪
                opts = get_optical_flow(frames)
                bbox, tubes = find_target_bbox(bboxes, opts)
        
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
    
    for video_file in tqdm(video_files, desc="处理视频"):
        # 获取 clip ID（文件名，不含扩展名）
        clip_id = video_file.stem
        
        # 生成 bbox
        bbox = get_bbox_for_clip(str(video_file), args.yolo_model)
        
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

