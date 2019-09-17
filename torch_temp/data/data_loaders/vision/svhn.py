from torchvision import datasets, transforms
from torch_temp.data.data_loaders.base import BaseDataLoader


class SVHNLoader(BaseDataLoader):
    """SVHN data loading + input transformations"""

    def __init__(self, data_dir, batch_size, shuffle,
                 validation_split=0.0,
                 num_workers=1,
                 training=True):

        if training is True:
            split = 'train'
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            split = 'test'
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.data_dir = data_dir
        self.dataset = datasets.SVHN(self.data_dir,
                                     split=split,
                                     download=True,
                                     transform=trans)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)
