import torch.nn as nn
from torch_temp.model.loss.base import BaseLoss


class TorchLoss(BaseLoss):
    def __init__(self, l_name, bargs, kwargs):
        super().__init__(*bargs)
        self.tloss = getattr(nn, l_name)(**kwargs)

    def forward(self, output, targets):
        return self.tloss(output, targets)
