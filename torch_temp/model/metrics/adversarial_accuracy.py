import torch
from .base import BaseMetric


class AdversarialAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversarial = True

    def forward(self, output, target, *args, **kwargs):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()

        return correct / len(target)

    def get_name(self):
        return "AdversarialAccuracy"
