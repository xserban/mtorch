import torch
from .base import BaseScheduler


class TorchScheduler(BaseScheduler):
    def __init__(self, optimizer, name, priority=0, active=False, *args, **kwargs):
        super().__init__(active, priority)
        self.optimizer = optimizer
        self.scheduler = getattr(torch.optim.lr_scheduler,
                                 name)(optimizer=optimizer, *args, **kwargs)

    def step(self, epoch):
        self.scheduler.step()
