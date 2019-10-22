from torch.nn.modules.loss import NLLLoss
from .base import BaseLoss


class NLL(NLLLoss, BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
