from .base import BaseLoss
import torch.nn.functional as F


class CosineSimilarityLoss(BaseLoss):
    def __init__(self, dim=1, eps=1e-8, reduction="mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        sim = 1 - F.cosine_similarity(output, target, self.dim, self.eps)
        if self.reduction == "mean":
            return sim.mean()
