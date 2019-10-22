from .base import BaseTransformation
from torchvision import transforms


class CIFARTransformations(BaseTransformation):
    def __init__(self):
        pass

    def get_train_trans(self):
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
        return transforms.Compose(transf)

    def get_test_trans(self):
        transf = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
        return transforms.Compose(transf)
