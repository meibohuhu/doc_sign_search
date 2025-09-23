import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image        

class LLaVA(nn.Module):
    def __init__(self):
        super().__init__()
        
        model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def forward(self, images):
        # For LLaVA 1.5, we need to use the simpler prompt format
        prompt = "USER: <image>\nWhat do the ASL signs in this video mean? Translate the ASL signs to English. Provide the translation in one sentence.\nASSISTANT:"
        
        # Process each image with the same prompt
        results = []
        for image in images:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device, torch.float16)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # Extract only the assistant's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text.strip()
                
            results.append(response)
        
        return results