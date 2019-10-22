from abc import abstractmethod


class BaseTransformation(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_train_trans(self):
        raise NotImplementedError

    @abstractmethod
    def get_test_trans(self):
        raise NotImplementedError
