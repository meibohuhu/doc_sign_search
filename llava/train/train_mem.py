import sys
import os
# Add the parent directory of llava to make llava importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from train import train

if __name__ == "__main__":
    train()
