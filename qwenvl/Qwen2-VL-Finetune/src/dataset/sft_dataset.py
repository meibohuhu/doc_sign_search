import copy
import os
from pathlib import Path
from typing import Dict, Optional
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai, pad_sequence
### mh: 2025-11-15: add mask 
from .mask_utils import load_mask, prepare_mask_tensor

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        ### mh: 2025-11-15: add mask 
        self.video_root = Path(self.data_args.image_folder).expanduser() if self.data_args.image_folder else None

        self.mask_root = Path(data_args.mask_folder).expanduser() if data_args.mask_folder else None
        if data_args.mask_file_suffix:
            suffix = data_args.mask_file_suffix
            self.mask_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        else:
            self.mask_suffix = ".npz"
        self.mask_key = data_args.mask_key
        self.mask_dilation = data_args.mask_dilation
        self.mask_blur_kernel = data_args.mask_blur_kernel
        self.bg_noise_std = data_args.fbcf_bg_noise_std
        self.use_masks = self.mask_root is not None
        
        # Debug: Count how many samples have mask files (only check first 100 to avoid slowdown)
        if self.use_masks and self.mask_root:
            self._mask_stats_checked = False
        else:
            self._mask_stats_checked = True
# ###########################################
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # mh: 2025-10-11 Retry logic to skip corrupted/missing videos, consequentially corrupt videos will be skipped too
        max_retries = 10
        for attempt in range(max_retries):
            try:
                idx = (i + attempt) % len(self.list_data_dict)
                sample = self.list_data_dict[idx]
                sources = sample  ### mh: 2025-11-15: add mask 

                is_video = False
                ### mh: 2025-11-15: add mask 
                video_mask_paths = []
                ###########################################
                processor = self.processor
                if "image" in sources:
                    videos = None
                    grid_key = "image_grid_thw"
                    pixel_key = "pixel_values"

                    image_files = sources["image"]
                    image_folder = self.data_args.image_folder

                    if isinstance(image_files, str):
                        image_files = [image_files]

                    images = []

                    for image_file in image_files:
                        if not os.path.exists(image_file):
                            if not image_file.startswith("http"):
                                image_file = os.path.join(image_folder, image_file)
                        images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

                elif "video" in sources:
                    is_video = True
                    images=None
                    grid_key = "video_grid_thw"
                    pixel_key = "pixel_values_videos"

                    video_files = sources["video"]
                    video_folder = self.data_args.image_folder

                    ### mh: 2025-11-15: add mask 
                    video_mask_paths = []
                    ##########################################
                    if isinstance(video_files, str):
                        video_files = [video_files]

                    videos = []
                    
                    # mh: 2025-11-15: Debug: Check mask statistics on first call
                    if self.use_masks and not self._mask_stats_checked and i == 0 and attempt == 0:
                        self._mask_stats_checked = True
                        # Count masks for first 100 samples
                        mask_count = 0
                        total_checked = min(100, len(self.list_data_dict))
                        for check_idx in range(total_checked):
                            check_sample = self.list_data_dict[check_idx]
                            if "video" in check_sample:
                                check_videos = check_sample["video"]
                                if isinstance(check_videos, str):
                                    check_videos = [check_videos]
                                for check_video in check_videos:
                                    if not check_video.startswith("http") and not os.path.isabs(check_video):
                                        check_video_path = os.path.join(video_folder, check_video)
                                    else:
                                        check_video_path = check_video
                                    check_mask_path = self._resolve_mask_path(check_video_path)
                                    if check_mask_path and os.path.exists(check_mask_path):
                                        mask_count += 1
                        coverage = 100 * mask_count / total_checked
                        # print(f"📊 Mask coverage: {mask_count}/{total_checked} ({coverage:.1f}%)")
                        if mask_count == 0:
                            print(f"  ⚠️  WARNING: No mask files found! FBCF will not be applied.")
                        elif mask_count < total_checked * 0.5:
                            print(f"  ⚠️  WARNING: Less than 50% of videos have masks. FBCF will only apply to videos with masks.")
                    ###########################################
                    for video_file in video_files:
                        # Handle different path scenarios
                        if not video_file.startswith("http") and not os.path.isabs(video_file):
                            # First, try direct path in video_folder
                            direct_path = os.path.join(video_folder, video_file)
                            if os.path.exists(direct_path):
                                video_file = direct_path
                            # If not found and there's an 'id' field, try using id as subdirectory
                            elif "id" in sources:
                                subdir_path = os.path.join(video_folder, sources["id"], video_file)
                                if os.path.exists(subdir_path):
                                    video_file = subdir_path
                                else:
                                    # Use direct path anyway (will fail gracefully in get_video_info)
                                    video_file = direct_path
                            else:
                                video_file = direct_path
                        
                        # Check if file exists before trying to load
                        if not os.path.exists(video_file) and not video_file.startswith("http"):
                            raise FileNotFoundError(f"Video file not found: {video_file}")
                        
                        video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.fps, self.nframes)
                        videos.append(video_input)
                         ### mh: 2025-11-15: add mask 
                        if self.use_masks:
                            resolved_mask_path = self._resolve_mask_path(video_file)
                            video_mask_paths.append(resolved_mask_path)
                else:
                    grid_key = None
                    pixel_key = None
                    images=None
                    videos=None

                ### mh: 2025-11-15: add mask 
                ### sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))
                sources = copy.deepcopy(llava_to_openai(sample['conversations'], is_video=is_video))
                ###########################################

                all_input_ids = []
                all_labels = []
                all_pixel_values = []
                 ### mh: 2025-11-15: add fg and bg pixel values 
                all_pixel_values_fg = []
                all_pixel_values_bg = []
                all_image_grid_thw = []
                all_second_gird = []
                ###########################################
                image_curr_count = 0
                video_curr_count = 0
                # ===========================================
                # mh: 2025-10-08
                # Extract system message from conversations if present   
                system_message_from_conversation = None
                if sources and sources[0]['role'] == 'system':
                    system_message_from_conversation = sources[0]['content']
                    sources = sources[1:]  # Remove system message from sources
                
                # Add system message (either from conversation or default)
                if system_message_from_conversation:
                    # Use system message from conversation
                    system_message = f"{DEFAULT_IM_START_TOKEN}system\n{system_message_from_conversation}{DEFAULT_IM_END_TOKEN}\n"
                elif len(SYSTEM_MESSAGE) > 0:
                    # Use default system message
                    system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
                else:
                    system_message = None
                    
                if system_message:
                    system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
                    system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

                    all_input_ids.append(system_message_input_ids.squeeze(0))
                    all_labels.append(system_labels.squeeze(0))

                for _, j in enumerate(range(0, len(sources), 2)):
                    user_input = sources[j]
                    gpt_response = sources[j + 1]

                    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

                    if DEFAULT_IMAGE_TOKEN in user_input:
                        num_images = user_input.count(DEFAULT_IMAGE_TOKEN)
                        # Slice the images list to get the images for the current turn.
                        images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                        inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                        prompt_input_ids = inputs['input_ids']
                        all_pixel_values.append(inputs[pixel_key])
                        all_image_grid_thw.append(inputs[grid_key])
                        image_curr_count += num_images

                    elif DEFAULT_VIDEO_TOKEN in user_input:
                        num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                        # Slice the videos list to get the videos for the current turn.
                        videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                        if "Qwen2.5" in self.model_id:
                            inputs = processor(text=[user_input], images=images, videos=videos_for_this_turn, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                            all_second_gird.extend(inputs["second_per_grid_ts"])
                        else:
                            inputs = processor(text=[user_input], images=images, videos=videos_for_this_turn, padding=False, do_resize=False, return_tensors='pt')
                        prompt_input_ids = inputs['input_ids']

                        ### mh: 2025-11-15: add fg and bg pixel values 
                        ## all_pixel_values.append(inputs[pixel_key])
                        video_pixels = inputs[pixel_key]
                        fg_pixels = None
                        bg_pixels = None
                        if self.use_masks and len(video_mask_paths) >= video_curr_count + num_videos:
                            mask_slice = video_mask_paths[video_curr_count : video_curr_count + num_videos]
                            # Get video_grid_thw to extract frame information
                            video_grid_thw = inputs[grid_key]  # shape: [num_videos, 3] where 3 = [T, H, W]
                            mask_tensor = self._build_video_mask_tensor(mask_slice, video_pixels, video_grid_thw)
                            if mask_tensor is not None:
                                # mask_tensor shape: [sum(T*H*W), 1] (flattened to match video_pixels)
                                # video_pixels shape: [num_patches, feature_dim] 
                                # We need to ensure they can be broadcast together
                                # Expand mask to match video_pixels feature dimension if needed
                                if mask_tensor.shape[0] == video_pixels.shape[0]:
                                    # Expand mask from [N, 1] to [N, feature_dim] to match video_pixels
                                    if mask_tensor.dim() == 2 and mask_tensor.shape[1] == 1:
                                        mask_tensor = mask_tensor.expand(-1, video_pixels.shape[1])
                                    fg_pixels = video_pixels * mask_tensor  ## 前景：mask区域保留
                                    bg_pixels = video_pixels * (1 - mask_tensor) ## 背景：mask区域置0
                                else:
                                    print(f"⚠️  WARNING: Shape mismatch - mask_tensor: {mask_tensor.shape}, video_pixels: {video_pixels.shape}")
                                    fg_pixels = None
                                    bg_pixels = None
                                    
                                # Apply background noise if enabled and bg_pixels was created
                                if bg_pixels is not None and self.bg_noise_std > 0:
                                    ## addd noise to bg pixels 背景添加噪声
                                    noise = torch.randn_like(video_pixels) * self.bg_noise_std
                                    bg_pixels = bg_pixels + noise * (1 - mask_tensor)
                        all_pixel_values.append(video_pixels) # 原始video
                        if fg_pixels is not None and bg_pixels is not None:
                            all_pixel_values_fg.append(fg_pixels)  # 前景video  
                            all_pixel_values_bg.append(bg_pixels)  # 背景video
                        ###########################################
                        all_image_grid_thw.append(inputs[grid_key])
                        video_curr_count += num_videos
               
                    else:
                        prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                    response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                    input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                    labels = torch.cat(
                        [
                            torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                            response_input_ids.squeeze(0),
                        ],
                        dim=0,
                    )

                    all_input_ids.append(input_ids)
                    all_labels.append(labels)

                # There is no need for eos or bos tokens in the input_ids
                # Qwen2-VL does not use them
                input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
                labels = torch.cat(all_labels, dim=0).to(torch.long)

                # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
                # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

                attention_mask = (input_ids > -1000000).to(torch.long)

                data_dict = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                if pixel_key and grid_key:
                    pixel_values = torch.cat(all_pixel_values, dim=0)
                    image_thw = torch.cat(all_image_grid_thw, dim=0)
                    data_dict[pixel_key] = pixel_values
                    data_dict[grid_key] = image_thw

                    ## mh: 2025-11-15: add fg and bg pixel values
                    if pixel_key == "pixel_values_videos" and len(all_pixel_values_fg) > 0:
                        pixel_values_fg = torch.cat(all_pixel_values_fg, dim=0)
                        pixel_values_bg = torch.cat(all_pixel_values_bg, dim=0)
                        data_dict["pixel_values_videos_fg"] = pixel_values_fg
                        data_dict["pixel_values_videos_bg"] = pixel_values_bg
                    ###########################################
                if len(all_second_gird) > 0:
                    second_gird = all_second_gird
                    data_dict["second_per_grid_ts"] = second_gird

                return data_dict
            
            except Exception as e:
                # Log the error and try the next sample
                video_file_info = sample.get('video', 'unknown') if isinstance(sample, dict) else 'unknown'
                print(f"⚠️  Warning: Failed to load sample {idx} (video: {video_file_info}). Error: {str(e)[:200]}")
                
                if attempt < max_retries - 1:
                    continue  # Try next sample
                else:
                    # If all retries failed, raise the error
                    raise RuntimeError(f"Failed to load valid sample after {max_retries} attempts. Last error: {str(e)}")

## mh: 2025-11-15: add mask path resolution
    def _resolve_mask_path(self, video_file: str) -> Optional[str]:
        if self.mask_root is None:
            return None
        video_path = Path(video_file)
        rel_path = None
        if self.video_root:
            try:
                rel_path = video_path.relative_to(self.video_root)
            except ValueError:
                rel_path = Path(video_path.name)
        elif video_path.is_absolute():
            rel_path = Path(video_path.name)
        else:
            rel_path = video_path
        rel_path = rel_path.with_suffix(self.mask_suffix)
        return str(self.mask_root / rel_path)

    def _build_video_mask_tensor(self, mask_paths, video_tensor: torch.Tensor, video_grid_thw: torch.Tensor) -> Optional[torch.Tensor]:
        if not mask_paths or any(path is None for path in mask_paths):
            return None
        
        # video_tensor shape is [num_patches, feature_dim] (flattened)
        # video_grid_thw shape is [num_videos, 3] where each row is [T, H, W]
        # We need to process each video's mask separately since they may have different frame counts
        if video_grid_thw.dim() != 2 or video_grid_thw.shape[1] != 3:
            print(f"⚠️  WARNING: Unexpected video_grid_thw shape: {video_grid_thw.shape}")
            return None
        
        if len(mask_paths) != video_grid_thw.shape[0]:
            print(f"⚠️  WARNING: Number of mask paths ({len(mask_paths)}) != number of videos ({video_grid_thw.shape[0]})")
            return None
        
        mask_tensors = []
        for idx, mask_path in enumerate(mask_paths):
            # Get frame count and dimensions for this specific video
            num_frames = int(video_grid_thw[idx, 0].item())  # T (frames)
            height = int(video_grid_thw[idx, 1].item())      # H
            width = int(video_grid_thw[idx, 2].item())       # W
            
            mask_np = load_mask(mask_path, self.mask_key)
            if mask_np is None:
                if mask_path and not os.path.exists(mask_path):
                    print(f"⚠️  WARNING: Mask file not found: {mask_path}")
                return None
            
            mask_tensor = prepare_mask_tensor(
                mask_np,
                target_frames=num_frames,
                target_height=height,
                target_width=width,
                dilation=self.mask_dilation,
                blur_kernel=self.mask_blur_kernel,
            )
            # mask_tensor shape: [T, 1, H, W]
            # We need to flatten it to match video_tensor structure
            # After flattening: [T * H * W, 1]
            T, C, H, W = mask_tensor.shape
            mask_tensor_flat = mask_tensor.view(T * H * W, C)  # [T*H*W, 1]
            mask_tensors.append(mask_tensor_flat)
        
        # Concatenate masks for all videos
        mask_stack = torch.cat(mask_tensors, dim=0)  # [sum(T*H*W), 1]
        mask_stack = mask_stack.to(video_tensor.device, dtype=video_tensor.dtype)
        
        return mask_stack
###########################################
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        
        ## mh: 2025-11-15: add fg and bg pixel values
        batch_pixel_video_fg_values = []
        batch_pixel_video_bg_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        ###########################################
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            ## mh: 2025-11-15: add fg and bg pixel values
            if "pixel_values_videos_fg" in keys:
                batch_pixel_video_fg_values.append(example["pixel_values_videos_fg"])
            if "pixel_values_videos_bg" in keys:
                batch_pixel_video_bg_values.append(example["pixel_values_videos_bg"])
            ###########################################
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw
        ### mh: 2025-11-15: add fg and bg pixel values
        if len(batch_pixel_video_fg_values) > 0:
            if len(batch_pixel_video_fg_values) != len(batch_pixel_video_values):
                print(f"⚠️  WARNING: fg_pixels count ({len(batch_pixel_video_fg_values)}) != original video count ({len(batch_pixel_video_values)})")
            data_dict["pixel_values_videos_fg"] = torch.cat(batch_pixel_video_fg_values, dim=0)
        if len(batch_pixel_video_bg_values) > 0:
            if len(batch_pixel_video_bg_values) != len(batch_pixel_video_values):
                print(f"⚠️  WARNING: bg_pixels count ({len(batch_pixel_video_bg_values)}) != original video count ({len(batch_pixel_video_values)})")
            data_dict["pixel_values_videos_bg"] = torch.cat(batch_pixel_video_bg_values, dim=0)
        ###########################################
        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)