import torch.nn as nn
from torch_temp.model.blocks.wideresnet import WideResNet


class WideResNetWrapper(WideResNet):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super().__init__(depth, num_classes, widen_factor=widen_factor, dropRate=dropRate)
