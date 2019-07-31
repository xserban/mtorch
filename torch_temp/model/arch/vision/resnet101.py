import torch.nn as nn
from torch_temp.model.blocks.resnet import *


class ResNet101(ResNet):
    def __init__(self, num_channels=1, num_classes=10):
        super(ResNet101, self).__init__(Bottleneck, [
            3, 4, 23, 3], num_channels, num_classes)
