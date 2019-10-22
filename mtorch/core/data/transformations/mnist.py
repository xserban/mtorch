from .base import BaseTransformation
from torchvision import transforms


class MNISTTransformations(BaseTransformation):
    def __init__(self):
        pass

    def get_train_trans(self):
        transf = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        return transforms.Compose(transf)

    def get_test_trans(self):
        transf = [
            transforms.ToTensor(),
        ]
        return transforms.Compose(transf)
