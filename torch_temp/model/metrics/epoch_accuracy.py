import torch
from .base import BaseMetric


class EpochAccuracy(BaseMetric):

    def forward(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        # returning just the number of correct items
        # because they are divided in the trainer by the
        # total nr of items
        return correct / len(target)

    def get_name(self):
        return "EpochAccuracy"
