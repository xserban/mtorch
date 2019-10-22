from core.data.data_loaders.base import BaseDataLoader
from core.data.datasets.vision.miniimagenet import MiniImageNet


class MiniImageNetLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,
                 n_way, k_shot, k_query, resize,
                 shuffle=True, validation_split=0.0,
                 num_workers=1, training=True):
        print("[INFO][DATA] \t Preparing MiniImageNet Dataset")
        mode = "train" if training is True else "test"
        self.dataset = MiniImageNet(
            data_dir, mode, batch_size, n_way, k_shot, k_query, resize)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)
