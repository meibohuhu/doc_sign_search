import torch._dynamo
torch._dynamo.config.suppress_errors = True

from llava.train.train import train

if __name__ == "__main__":
    train()
