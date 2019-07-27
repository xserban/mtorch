from torchvision import datasets, transforms
from data.data_loaders.base import BaseDataLoader


class CIFAR10Loader(BaseDataLoader):
    """ CIFAR10 data loading + transformations """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        print('[INFO] Preparing Cifar10 dataset ...')
        if training is True:
            trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trans)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
