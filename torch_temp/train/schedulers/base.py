from abc import abstractmethod


class BaseScheduler:
    def __init__(self, priority=0, active=False):
        self.priority = priority
        self.active = False

    @abstractmethod
    def step(self, epoch):
        raise NotImplementedError

    def is_active(self):
        return self.active
