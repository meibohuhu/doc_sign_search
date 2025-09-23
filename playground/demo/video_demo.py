import argparse
import torch
import json
import os
import math

from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoConfig

import cv2
import base64  # only used by load_video_base64 if you keep it; safe to leave
from PIL import Image
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_anyres_image, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def resolve_image_size(model, args):
    """Choose vision image size via CLI > config.json > model.config > default=1024."""
    if getattr(args, "image_size", None):
        try:
            return int(args.image_size)
        except Exception:
            pass
    cfg_paths = []
    if getattr(args, "fastvit_config", None):
        cfg_paths.append(args.fastvit_config)
    if getattr(args, "model_path", None):
        cand = os.path.join(args.model_path, "config.json")
        if os.path.isfile(cand):
            cfg_paths.append(cand)
    for p in cfg_paths:
        if p and os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    cfg = json.load(f)
                sz = cfg.get("vision_tower_image_size") or cfg.get("image_cfg", {}).get("image_size")
                if sz:
                    return int(sz)
            except Exception:
                pass
    if hasattr(model, "config"):
        for k in ("vision_tower_image_size", "image_size"):
            v = getattr(model.config, k, None)
            if v:
                try: return int(v)
                except Exception: pass
        vc = getattr(model.config, "vision_config", None)
        if vc is not None and getattr(vc, "image_size", None):
            try: return int(vc.image_size)
            except Exception: pass
    return 1024


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to a video file or directory of videos.")
    parser.add_argument("--fastvit-config", type=str, default=None, help="Path to FastViT/MobileCLIP config.json.")
    parser.add_argument("--image_size", type=int, default=None, help="Force vision image size (overrides configs).")
    parser.add_argument("--output_dir", required=True, help="Directory to save results JSON.")
    parser.add_argument("--output_name", required=True, help="JSON filename (without extension).")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    return parser.parse_args()


def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    fps = max(round(vr.get_avg_fps()), 1)
    video_time = total_frame_num / max(vr.get_avg_fps(), 1e-6)
    frame_idx = [i for i in range(0, total_frame_num, fps)]
    frame_time = [i / max(vr.get_avg_fps(), 1e-6) for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = max(args.for_get_frames_num, 1)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / max(vr.get_avg_fps(), 1e-6) for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time_str, video_time


def run_inference(args):
    if args.model_path.strip().lower() == "gpt4v":
        raise ValueError("This script no longer supports OpenAI GPT-4V. Pass a local/HF model path for --model-path.")

    model_name = get_model_name_from_path(args.model_path)

    cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
    if args.overwrite:
        overwrite_config = {
            "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
            "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
            "mm_newline_position": args.mm_newline_position,
        }

        # Adjust rope scaling / max lengths if needed (non-Qwen case like Vicuna)
        if "qwen" not in args.model_path.lower():
            if hasattr(cfg_pretrained, "mm_vision_tower") and isinstance(cfg_pretrained.mm_vision_tower, str):
                vision_res = 224 if "224" in cfg_pretrained.mm_vision_tower else 336
            else:
                vision_res = 336
            least_token_number = args.for_get_frames_num * ( (16 if vision_res==224 else 24) // args.mm_spatial_pool_stride )**2 + 1000
            scaling_factor = math.ceil(least_token_number / 4096)
            if scaling_factor >= 2:
                if "vicuna" in getattr(cfg_pretrained, "_name_or_path", "").lower():
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name
        )

    image_size = resolve_image_size(model, args)
    print(f"[video_demo] Resolved vision image size: {image_size}")

    # Propagate config-driven flags (safe defaults if absent)
    args.force_sample = getattr(model.config, "force_sample", False)
    args.add_time_instruction = getattr(model.config, "add_time_instruction", False)

    os.makedirs(args.output_dir, exist_ok=True)
    answers_path = os.path.join(args.output_dir, f"{args.output_name}.json")

    # Collect video paths
    video_path = args.video_path
    all_video_paths = []
    if os.path.isdir(video_path):
        for filename in os.listdir(video_path):
            cur = os.path.join(video_path, filename)
            if os.path.isfile(cur):
                all_video_paths.append(cur)
    else:
        all_video_paths.append(video_path)

    with open(answers_path, "w", encoding="utf-8") as ans_file:
        for vp in all_video_paths:
            sample_set = {"Q": args.prompt, "video_name": vp}

            if not os.path.exists(vp):
                print(f"[warn] File not found: {vp}")
                continue

            # Load frames -> preprocess -> to CUDA
            frames, frame_time, video_time = load_video(vp, args)
            video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
            video_list = [video_tensor]  # LLaVA expects a list of tensors

            # Build prompt with image/video tokens
            qs = args.prompt or "Describe the video."
            if args.add_time_instruction:
                time_instruction = (
                    f"The video lasts for {video_time:.2f} seconds, and {len(video_list[0])} frames are uniformly sampled. "
                    f"These frames are located at {frame_time}. Please answer the following question."
                )
                qs = f"{time_instruction}\n{qs}"

            if getattr(model.config, "mm_use_im_start_end", False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            # Conversation template fallback
            conv_mode = args.conv_mode
            if conv_mode is None:
                name = (tokenizer.name_or_path or "").lower()
                if "qwen" in name:
                    conv_mode = "qwen_1_5" if "1.5" in name or "qwen2" not in name else "qwen_2"
                elif "mistral" in name:
                    conv_mode = "mistral_instruct"
                else:
                    conv_mode = "vicuna_v1"

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            # Ensure pad token for Qwen
            if tokenizer.pad_token_id is None and "qwen" in (tokenizer.name_or_path or "").lower():
                print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643

            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                # Deterministic decoding by default; tune as needed
                output_ids = model.generate(
                    inputs=input_ids,
                    images=video_list,
                    attention_mask=attention_masks,
                    modalities="video",
                    do_sample=False,
                    temperature=0.0,
                    top_p=0.1,
                    num_beams=1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_riteria := stopping_criteria],  # keep var named to avoid lints
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # Trim trailing stop token if present (non-Mistral branch in your original logic)
            if "mistral" not in getattr(cfg_pretrained, "_name_or_path", "").lower():
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            sample_set["pred"] = outputs
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()

            print(f"\nQuestion: {args.prompt}\n")
            print(f"Response: {outputs}\n")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
