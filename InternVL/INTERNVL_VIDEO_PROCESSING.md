视频文件 (96 frames)
 ↓
每帧 → PIL Image 
↓
预处理 (resize, normalize) → pixel_values.shape: torch.Size([96, 3, 224, 224])
 ↓
InternVisionModel (ViT)
 ├─ InternVisionEmbeddings
 │ ├─ Patch Embedding (Conv2d) → [96, 64, 2048]
 │ └─ Position Embedding
 ├─ InternVisionEncoder (多层Transformer)
 │ └─ 每层: Attention + MLP
 └─ 输出: [96, 257, 1024] encoder_outputs.last_hidden_state.shape: torch.Size([96, 257, 1024])
 ↓ 
移除CLS token 
↓
Reshape
 ↓
Pixel Shuffle (降采样) → [96, 64, 1408] vit_embeds.shape before mlp1: torch.Size([96, 64, 4096])
 ↓
MLP投影 → [96, 64, 4096] (LLM hidden_size) vit_embeds.shape after mlp1: torch.Size([96, 64, 2048])
 ↓		input_embeds.shape: torch.Size([7074, 2048]), vit_embeds.shape: torch.Size([96, 64, 2048])
展平 → [7074, 2048] (60帧 × 64 tokens)
 ↓
替换文本中的 <IMG_CONTEXT> tokens
 ↓
传入LLM进行生成
