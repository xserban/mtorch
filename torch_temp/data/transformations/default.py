from .base import BaseTransformation
from torchvision import transforms


class DefaultTransformations(BaseTransformation):
    def __init__(self):
        pass

    def get_train_trans(self):
        transf = [transforms.ToTensor()]
        return transforms.Compose(transf)

    def get_test_trans(self):
        return self.get_train_trans()
