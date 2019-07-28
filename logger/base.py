from abc import abstractmethod


class BaseLogger(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def log_batch(self, step, env, loss, custom_metrics):
        raise NotImplementedError

    @abstractmethod
    def log_custom_metrics(self, metrics):
        raise NotImplementedError

    @abstractmethod
    def log_epoch(self, epoch, loss, metrics):
        raise NotImplementedError
