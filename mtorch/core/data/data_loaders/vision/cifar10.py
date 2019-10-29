from torchvision import datasets, transforms
from core.data.data_loaders.base import BaseDataLoader


class CIFAR10Loader(BaseDataLoader):
    """ CIFAR10 data loading + transformations """

    def __init__(self, data_dir,
                 batch_size=128,
                 training=True,
                 validation_split=0.0,
                 shuffle=False,
                 transformations="DefaultTransformations",
                 **kwargs):
        print("[INFO][DATA] \t Preparing Cifar10 dataset ...")
        _transf = BaseDataLoader.get_transformations(
            self, name=transformations)

        self.trans = _transf.get_train_trans() if training is True \
            else _transf.get_test_trans()

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(
            self.data_dir, train=training, download=True, transform=self.trans)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, **kwargs)

    def get_class_names(self):
        return('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
