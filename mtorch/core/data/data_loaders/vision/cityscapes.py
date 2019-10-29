###
# !!! Pytorch requires to manuual download the CityScapes dataset
# and add it to the data_dir
###
from torchvision import datasets, transforms
from core.data.data_loaders.base import BaseDataLoader


class CityscapesLoader(BaseDataLoader):
    """Cityscapes data loading"""

    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0,
                 split="train",
                 transformations="DefaultTransformations",
                 kwargs={}):
        print("[INFO][DATA] \t Preparing the CityScapes dataset ...")

        _transf = BaseDataLoader.get_transformations(
            self, name=transformations)

        self.trans = _transf.get_train_trans() if split == "train" \
            else _transf.get_test_trans()

        self.data_dir = data_dir
        self.dataset = datasets.Cityscapes(
            self.data_dir, split=split, transform=self.trans, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split)
