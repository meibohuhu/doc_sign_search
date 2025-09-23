# mhu update 09/22/2025 add evaluation metrics integration
import os
os.environ["DISABLE_FLASH_ATTN"] = "1"

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
import argparse
import json
from tqdm import tqdm
import math
import random
random.seed(10)
warnings.filterwarnings("ignore")
# Load the OneVision model
# import sys
# sys.path.insert(0,"/home/vp1837/ASL-AI/ASL_research/LLaVA-NeXT/")

import os, json  # make sure these imports exist

# Import evaluation metrics
import sys
sys.path.append('/local1/mhu/LLaVANeXT_RC/evaluation/uni_sign')
from SLRT_metrics import translation_performance


### mhu update 09/22/2025 comment: for base model: python simpleQA_metrics.py --model-path lmms-lab/llava-onevision-qwen2-0.5b-ov --image_size 336
def resolve_image_size(model, args):
    """
    Pick image size from (1) --image_size,
    (2) the FastViT/MobileCLIP config.json,
    (3) model.config fallbacks,
    else default=1024.
    """
    # 1) explicit CLI
    if getattr(args, "image_size", None):
        try:
            return int(args.image_size)
        except Exception:
            pass

    # 2) fastvit config file
    cfg_paths = []
    if getattr(args, "fastvit_config", None):
        cfg_paths.append(args.fastvit_config)
    # also try model_path/config.json next to the merged checkpoint
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

    # 3) model.config fallbacks
    if hasattr(model, "config"):
        for k in ("vision_tower_image_size", "image_size"):
            v = getattr(model.config, k, None)
            if v:
                try:
                    return int(v)
                except Exception:
                    pass
        vc = getattr(model.config, "vision_config", None)
        if vc is not None and getattr(vc, "image_size", None):
            try:
                return int(vc.image_size)
            except Exception:
                pass

    # 4) default for MobileCLIP-L/1024
    return 1024


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
# Function to extract frames from video
def load_frames(video_file, num_frames_to_sample=10):
    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if
                   os.path.isfile(os.path.join(video_file, f))]
    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
    total_frames = len(frame_files)
    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

    # Read and store the sampled frames
    video = []
    for idx in sampled_indices:
        frame_path = frame_files[idx]
        try:
            with Image.open(frame_path) as img:
                frame = img.convert("RGB")
                video.append(frame)
        except IOError:
            rank0_print(f"Failed to read frame at path: {frame_path}")
    return video

## mhu update 09/19/2025 comment: it extract frames from video
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames

def eval_model(args):

    device = "cuda"
    device_map = "auto"
    model_path = os.path.expanduser(args.model_path)


    ###  mhu update 09/19/2025 Disable flash attention to avoid compatibility issues
    ## tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, args.model_base, args.model_name, device_map=device_map,  attn_implementation="eager")
    os.environ["DISABLE_FLASH_ATTN"] = "1"
    
    # Auto-detect model name if not provided
    if not hasattr(args, 'model_name') or not args.model_name:
        args.model_name = get_model_name_from_path(model_path)
    
    try:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path, args.model_base, args.model_name, 
            device_map=device_map, attn_implementation="eager"
        )
    except ValueError as e:
        if "image_newline" in str(e) or "shape" in str(e):
            print("⚠️  Model configuration mismatch. Trying without model_name specification...")
            # Try with auto-detected model name
            model_name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                model_path, args.model_base, model_name, 
                device_map=device_map, attn_implementation="eager"
            )
        else:
            raise e
############################
    image_size = resolve_image_size(model, args)
    print(f"[simpleQA_metrics] Resolved vision image size: {image_size}")

    model.eval()
    data_dict = json.load(open(args.question_file,'r'))

    data_dict = get_chunk(data_dict, args.num_chunks, args.chunk_idx)
    
    # Limit the number of samples if specified
    if args.max_samples is not None:
        original_size = len(data_dict)
        data_dict = data_dict[:args.max_samples]
        rank0_print(f"Limited to {len(data_dict)} samples (from {original_size} total)")
    
    rank0_print("chunk size:",len(data_dict))
    if not os.path.exists(args.out_dir):
    # If it doesn't exist, create the directory
        os.makedirs(args.out_dir)
    #out_file = open(os.path.join(args.out_dir,args.answers_file), 'a',encoding='utf-8')


    results=[]
    # Lists to store references and predictions for evaluation
    references = []
    predictions = []
    
    # we only use single image and video data for preference data generation
    for source in tqdm(data_dict):
        video_file=None
        image_file=None
        token_len=0
        try:
            if 'video' in source:
                video_file = source["video"]
                video = os.path.join(args.video_folder, video_file)
                video_frames = extract_frames(video, 32)
                # if not video_frames:
                #     print(f"Warning: No frames extracted from: {video}")
                #     results.append({
                #         "video": video_file,
                #         "prompt": source.get("conversations", [{}])[0].get("value", ""),
                #         "model_output": "ERROR: No frames extracted"
                #     })
                #     continue
                    
                image_tensors = process_images(video_frames, image_processor, model.config)
                image_sizes = [frame.size for frame in video_frames]
            elif 'image' in source:
                image_file = source["image"]
                if type(image_file) is list:
                    imgs=[Image.open(os.path.join(args.image_folder, img_f)).convert("RGB") for img_f in image_file]

                    # if len(image_file) > 1:
                    #     continue
                    #else:
                    image_sizes = [img.size for img in imgs]
                    image_tensors = process_images(imgs,image_processor, model.config)
                else:
                    img = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                    image_sizes = [img.size]
                    image_tensors = process_images([img], image_processor, model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
            conv_template = "qwen_1_5"
            # fq=source['conversations'][0]['value'].replace('<image>\n','')
            # fq=fq.replace('\n<image>','')
            # fq+="Generate the answer directly in one sentence."


            # fq = "What do the ASL signs in this video mean? Provide your translation in one clear sentence, If you cannot understand the signs, respond with 'I don't know'."            # fq = "Translate the American Sign Language (ASL) signs from gesture in this video to English. If you can understand the signs, respond with one clear sentence. If you cannot understand the signs, respond with 'don't know'."
            fq = "Translate the American Sign Language (ASL) signs in this video to English. Provide your translation in one clear sentence. If you cannot understand the signs, respond with 'I don't know'."

            # fq = "Translate the American Sign Language (ASL) signs in this video to English. Provide an accurate translation that captures the meaning of each sign and the complete message being conveyed."

            fqs = f"{DEFAULT_IMAGE_TOKEN}\n{fq}"
            first_answer=source['conversations'][1]['value']
            conv = copy.deepcopy(conv_templates[conv_template])
            #conv.clear_message()


            conv.append_message(conv.roles[0], fqs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            #rank0_print(prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
                0).to(device)
            
            # mhu update 09/19/2025 Create attention mask to avoid warning
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)

            questions = []
            answers = []

            cont = model.generate(
                input_ids,
                attention_mask=attention_mask,  ## mhu update 09/19/2025 add attention_mask
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=128,
                top_k=None,
                top_p=None,
                use_cache=True,
            )

            output= tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            print("ground truth:",first_answer)
            print("Generation: ",output)
            
            # Store references and predictions for evaluation
            references.append(first_answer)
            predictions.append(output)
            
            results.append({
                "video": video_file,
                "prompt": fq,
                "model_output": output,
                "ground_truth": first_answer
            })
        except Exception as e:
            results.append({
                "video": source.get("video"),
                "prompt": source.get("conversations", [{}])[0].get("value"),
                "model_output": "ERROR"
            })
            continue
    #output_file = "./qwen_test7B.json"
    #print(f"\n Saving results to {output_file}...")
    with open(os.path.join(args.out_dir,args.answers_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(" All outputs saved successfully!")
    
    # Calculate evaluation metrics
    if args.enable_evaluation and references and predictions:
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        try:
            bleu_dict, rouge_score = translation_performance(references, predictions)
            
            print(f"\nBLEU Scores:")
            for k, v in bleu_dict.items():
                print(f"  {k}: {v:.4f}")
            
            print(f"\nROUGE-L F1 Score: {rouge_score:.4f}")
            
            # Save evaluation metrics to a separate file
            eval_results = {
                "bleu_scores": bleu_dict,
                "rouge_l_f1": rouge_score,
                "total_samples": len(references),
                "evaluation_summary": {
                    "bleu_1": bleu_dict.get("bleu1", 0),
                    "bleu_2": bleu_dict.get("bleu2", 0),
                    "bleu_3": bleu_dict.get("bleu3", 0),
                    "bleu_4": bleu_dict.get("bleu4", 0),
                    "rouge_l_f1": rouge_score
                }
            }
            
            eval_file = os.path.join(args.out_dir, "evaluation_metrics.json")
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            
            print(f"\nEvaluation metrics saved to: {eval_file}")
            
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            print("Continuing without evaluation metrics...")
    else:
        print("No valid references and predictions found for evaluation.")
    #     if video_file:
    #         visual_modality="video"
    #         visual_file=video_file
    #         token_len+=196*len(video_frames)*2
    #     else:
    #         visual_modality="image"
    #         visual_file=image_file
    #         token_len+=7290*2
    #     out_p={"id": source['id'],
    #            visual_modality: visual_file,
    #            "sampler": [fqs,first_answer],
    #            "questions": questions,
    #            "answers":answers,
    #            "token_len":token_len,}
    #
    #     try:
    #         out_file.write(json.dumps(out_p)+'\n')
    #         out_file.flush()
    #     except:
    #         pass
    # out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastvit-config", type=str, default=None,
                    help="Path to FastViT/MobileCLIP config.json (has vision_tower_image_size).")
    parser.add_argument("--image_size", type=int, default=None,
                    help="Force vision image size (overrides configs).")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--n_shot", type=int, default=2)
    parser.add_argument("--sampleNum", type=int, default=10000000)
    parser.add_argument("--enable_evaluation", action="store_true", default=True,
                       help="Enable evaluation metrics calculation (default: True)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: process all)")
    args = parser.parse_args()

    eval_model(args)
