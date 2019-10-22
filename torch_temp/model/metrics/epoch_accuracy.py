import torch
from .base import BaseMetric


class EpochAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, output, targets, *args, **kwargs):
        assert output.size(0) == targets.size(0)
        with torch.no_grad():
            _, pred = output.max(1)
            correct = pred.eq(targets).sum().item()

        return 100*(correct / targets.size(0))

    def get_name(self):
        return "EpochAccuracy"
