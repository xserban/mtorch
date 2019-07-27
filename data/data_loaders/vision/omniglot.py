from data.data_loaders.base import BaseDataLoader
from data.datasets.vision.omniglot import Omniglot


class OmniglotLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, transforms=None, target_transforms=None, download=False):
        print('[INFO] Preparing Omniglot Dataset')
        self.dataset = Omniglot(data_dir, transforms, target_transforms, download)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
