import torch
from llava.model.llava_arch import LlavaMetaModel

# Just instantiating the model class to test
model = LlavaMetaModel()

# Check if the image_newline parameter is set and has the expected shape
if hasattr(model, "image_newline"):
    print("image_newline shape:", model.image_newline.shape)
else:
    print("image_newline is not defined in LlavaMetaModel")
