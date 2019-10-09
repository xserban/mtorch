from .base import BaseTransformation
from torchvision import transforms


class ImageNetTransformations(BaseTransformation):
    def __init__(self):
        pass

    def get_train_trans(self):
        transf = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        return transforms.Compose(transf)

    def get_test_trans(self):
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        return transforms.Compose(transf)
