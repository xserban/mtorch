import torch
from .base import BaseMetric


class EpochAccuracy(BaseMetric):

    def forward(self, output, target):
        with torch.no_grad():
            _, pred = torch.max(output.data, 1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += (pred == target).sum().item()
        # returning just the number of correct items
        # because they are divided in the trainer by the
        # total nr of items
        return correct

    def get_name(self):
        return "EpochAccuracy"
