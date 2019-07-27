from torchvision import datasets, transforms
from data.data_loaders.base import BaseDataLoader

print(datasets.Omniglot)


class OmniglotLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, training=False, validation_split=0.0, num_workers=1):
        print('[INFO] Preparing Omniglot Dataset')
        # TODO: Fill in transformations
        if training is True:
            trans = transforms.Compose([])
        else:
            trans = transforms.Compose([])

        self.data_dir = data_dir
        self.dataset = datasets.Omniglot(self.data_dir, download=True, transform=trans)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
