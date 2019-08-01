from abc import abstractmethod


class BaseLoss:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self):
        raise NotImplementedError
