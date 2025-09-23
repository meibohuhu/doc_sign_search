# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time

#  7B "Video-Only" 
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"


tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, torch_dtype=torch.float16, device_map=device_map
)
model.eval()

video_path = "/home/vp1837/test/raw_videos/_G0RrDVpOZ4-5-rgb_front.mp4"


max_frames_num = 64

video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
video = [video]

conv_template = "qwen_1_5"  

time_instruciton = (
    f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
    f"These frames are located at {frame_time}.Please answer the following questions related to this video."
)

question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n" \
           "The given video is an ASL signing conversation by a person done using hand, face gestures and finger spelling. " \
           "Studying the continuous signing in the video with the help of video frames to understand each hand sign and facial expression " \
           "derive the coherent english translation of the signed words used in the video. Give the entire english translation of the video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

cont = model.generate(
    input_ids,
    images=video,
    modalities=["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)

text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
