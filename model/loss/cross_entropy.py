import torch.nn.functional as F
from .base import BaseLoss


class CrossEntropy(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, output, target):
        return F.cross_entropy(output, target)
