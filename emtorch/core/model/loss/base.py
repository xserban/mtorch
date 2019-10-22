from abc import abstractmethod

from torch.nn import Module


class BaseLoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
