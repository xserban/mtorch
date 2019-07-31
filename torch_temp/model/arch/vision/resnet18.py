import torch.nn as nn
from torch_temp.model.blocks.resnet import *


class ResNet18(ResNet):
    def __init__(self, num_channels=1, num_classes=10):
        super(ResNet18, self).__init__(BasicBlock, [
            2, 2, 2, 2], num_channels, num_classes)
