from torchvision import datasets, transforms
from torch_temp.data.data_loaders.base import BaseDataLoader


class MnistLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0,
                 training=True,
                 transformations="MNISTTransformations",
                 **kwargs):

        _transf = BaseDataLoader.get_transformations(
            self, name=transformations)

        trans = _transf.get_train_trans() if training is True \
            else _transf.get_test_trans()

        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trans)
        super().__init__(self.dataset, batch_size,
                         shuffle,
                         validation_split,
                         **kwargs)
