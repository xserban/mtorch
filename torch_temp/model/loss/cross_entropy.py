from torch.nn.modules.loss import CrossEntropyLoss
from .base import BaseLoss


class CrossEntropy(CrossEntropyLoss, BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
