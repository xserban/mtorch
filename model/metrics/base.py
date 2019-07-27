class BaseMetric(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def get_name(self):
        raise NotImplementedError
