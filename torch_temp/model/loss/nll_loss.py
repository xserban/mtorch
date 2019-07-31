import torch.nn.functional as F
from .base import BaseLoss


class NLL(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, output, target):
        return F.nll_loss(output, target)
