import cv2
import os
import glob
from collections import Counter

# folder = '/local1/mhu/sign_language_llm/how2sign/video/train_videos_raw'
folder = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips'
videos = glob.glob(os.path.join(folder, '*.mp4'))

print(f'Found {len(videos)} videos')
print('Checking FPS...\n')

fps_list = []
fps_dict = {}

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps > 0:
        fps_rounded = round(fps, 2)
        fps_list.append(fps_rounded)
        video_name = os.path.basename(video_path)
        fps_dict[video_name] = fps_rounded

# Count FPS distribution
fps_counter = Counter(fps_list)

print("=" * 50)
print("FPS Distribution:")
print("=" * 50)
for fps_value in sorted(fps_counter.keys()):
    count = fps_counter[fps_value]
    percentage = (count / len(videos)) * 100
    print(f"{fps_value:6.2f} fps: {count:4d} videos ({percentage:5.2f}%)")

print("\n" + "=" * 50)
print(f"Total videos: {len(videos)}")
print(f"Unique FPS values: {len(fps_counter)}")
print("=" * 50)

# Show some examples
print("\nSample videos with different FPS:")
print("-" * 50)
for fps_value in sorted(fps_counter.keys()):
    examples = [name for name, f in fps_dict.items() if f == fps_value][:3]
    print(f"\n{fps_value:.2f} fps examples:")
    for ex in examples:
        print(f"  - {ex}")

