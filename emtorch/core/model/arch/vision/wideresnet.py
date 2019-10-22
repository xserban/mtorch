import torch.nn as nn
from core.model.blocks.wideresnet import WideResNet


class WideResNetWrapper(WideResNet):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super().__init__(depth, num_classes,
                         widen_factor=widen_factor, dropout_rate=dropout_rate)
