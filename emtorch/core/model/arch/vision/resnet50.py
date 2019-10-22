import torch.nn as nn
from core.model.blocks.resnet import ResNet, Bottleneck


class ResNet50(ResNet):
    def __init__(self, num_channels=1, num_classes=10):
        super(ResNet50, self).__init__(Bottleneck, [
            3, 4, 6, 3], num_channels, num_classes)
