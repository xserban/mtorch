import torch
from .base import BaseMetric


class ProtoAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, output, target, *args, **kwargs):
        with torch.no_grad():
            assert output.shape[0] == len(target)
            correct = 0
            correct += torch.sum(output == target).item()

        return correct / len(target)

    def get_name(self):
        return "ProtoAccuracy"
