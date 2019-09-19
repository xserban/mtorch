import torch
from .base import BaseMetric


class TopkAccuracy(BaseMetric):
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def forward(self, output, target, *args, **kwargs):
        with torch.no_grad():
            pred = torch.topk(output, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)
