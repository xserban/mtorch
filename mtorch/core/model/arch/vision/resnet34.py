import torch.nn as nn
from core.model.blocks.resnet import ResNet, BasicBlock


class ResNet34(ResNet):
    def __init__(self, num_channels=1, num_classes=10):
        super(ResNet34, self).__init__(BasicBlock, [
            3, 4, 6, 3], num_channels, num_classes)
