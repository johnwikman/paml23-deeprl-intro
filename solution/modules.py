
import torch


class BytePixel2FloatPixel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float32) / 255.0
