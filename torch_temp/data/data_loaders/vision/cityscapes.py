from torchvision import datasets, transforms
from torch_temp.data.data_loaders.base import BaseDataLoader


class CityscapesLoader(BaseDataLoader):
    """Cityscapes data loading"""

    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 split="train",
                 kwargs={}):
        print("[INFO][DATA] \t Preparing Cifar10 dataset ...")
        if split == "train":
            trans = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.data_dir = data_dir
        self.dataset = datasets.Cityscapes(
            self.data_dir, split=split, transform=trans, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)
