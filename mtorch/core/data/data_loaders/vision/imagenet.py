from torchvision import datasets, transforms
from core.data.data_loaders.base import BaseDataLoader


class ImageNetLoader(BaseDataLoader):
    """ ImageNet data loading + transformations """

    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0,
                 training=True, transformations="DefaultTransformations",
                 **kwargs):
        print("[INFO][DATA] \t Preparing the ImageNet dataset ...")

        _transf = BaseDataLoader.get_transformations(
            self, name=transformations)
        trans = _transf.get_train_trans() if training is True \
            else _transf.get_test_trans()

        self.data_dir = data_dir
        split = "train" if training is True else "val"

        self.dataset = datasets.ImageNet(
            self.data_dir, split=split, download=True, transform=trans)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, **kwargs)
