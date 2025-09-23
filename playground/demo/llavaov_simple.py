import os
import torch
import torch.nn as nn
from PIL import Image
import copy


### mhu update 09/19/2025 object-oriented approach with error handling
## LLaVA class
## test_llavaov_simple.py call this class and load the model => run_llavov_simple.sh call this py

# Disable flash attention
os.environ["DISABLE_FLASH_ATTN"] = "1"

# Import the working LLaVA components
import sys
sys.path.insert(0, '/local1/mhu/LLaVANeXT_RC')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

class LLaVA(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use the stable 0.5B model
        model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        print(f"🔧 Loading LLaVA-OneVision 0.5B model: {model_path}")
        device = "cuda"
        device_map = "auto"
        
        # Auto-detect model name
        model_name = get_model_name_from_path(model_path)
        
        try:
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                model_path, None, model_name, 
                device_map=device_map, attn_implementation="eager"
            )
        except ValueError as e:
            if "image_newline" in str(e) or "shape" in str(e):
                print("⚠️  Model configuration mismatch. Trying without model_name specification...")
                model_name = get_model_name_from_path(model_path)
                self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                    model_path, None, model_name, 
                    device_map=device_map, attn_implementation="eager"
                )
            else:
                raise e
        
        self.model.eval()
        self.device = device
        print("✅ LLaVA-OneVision model loaded successfully")
        
    def forward(self, images):
        """
        Process a list of images and return ASL translations
        """
        results = []
        
        for image in images:
            try:
                # Process single image
                image_tensors = process_images([image], self.image_processor, self.model.config)
                image_sizes = [image.size]
                image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
                
                # Prepare conversation
                conv_template = "qwen_1_5"
                prompt = "What do the ASL signs in this video mean? Translate the ASL signs to English. Provide the translation in one sentence."
                fqs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
                
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], fqs)
                conv.append_message(conv.roles[1], None)
                prompt_formatted = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
                
                # Generate
                with torch.no_grad():
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        do_sample=True,
                        temperature=0.7,
                        max_new_tokens=128,
                        use_cache=True,
                    )
                
                output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                
                # Extract the response part (after the assistant marker)
                if "ASSISTANT:" in output:
                    response = output.split("ASSISTANT:")[-1].strip()
                elif "assistant" in output.lower():
                    response = output.split("assistant")[-1].strip()
                else:
                    response = output.strip()
                
                results.append(response)
                
            except Exception as e:
                print(f"❌ Error processing image: {e}")
                results.append(f"ERROR: {str(e)}")
        
        return results
