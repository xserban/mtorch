from torchvision import datasets, transforms
from core.data.data_loaders.base import BaseDataLoader


class CIFAR100Loader(BaseDataLoader):
    """ CIFAR100 data loading + transformations """

    def __init__(self, data_dir, batch_size,
                 shuffle=True, validation_split=0.0,
                 training=True,
                 transformations="DefaultTransformations",
                 **kwargs):
        print("[INFO][DATA] \t Preparing the CIFAR100 dataset ...")

        _transf = BaseDataLoader.get_transformations(
            self, name=transformations)

        self.trans = _transf.get_train_trans() if training is True \
            else _transf.get_test_trans()

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(
            self.data_dir, train=training, download=True, transform=self.trans)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split,
                         **kwargs)
