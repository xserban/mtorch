from torchvision import datasets, transforms
from torch_temp.data.data_loaders.base import BaseDataLoader


class ImageNetLoader(BaseDataLoader):
    """ ImageNet data loading + transformations """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        print('[INFO] Preparing the ImageNet dataset ...')
        if training is True:
            trans = transforms.Compose([
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])

        self.data_dir = data_dir
        if training is True:
            split = 'train'
        else:
            split = 'val'
        self.dataset = datasets.ImageNet(
            self.data_dir, split=split, download=True, transform=trans)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
