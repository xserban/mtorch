from core.data.data_loaders.base import BaseDataLoader
from core.data.datasets.vision import OmniglotDataset, OmniglotNShot


class OmniglotLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, tasks_num,
                 n_way, k_shot, k_query, imgsz,
                 shuffle=True, validation_split=0.0, num_workers=1):
        print("[INFO][DATA] \t Preparing Omniglot Dataset")

        self.data_dir = data_dir + "omniglot/"

        # self.dataset = datasets.Omniglot(
        #     self.data_dir, download=True, transform=trans)
        self.dataset = OmniglotDataset(
            self.data_dir, download=True)
        self.n_shot_dataset = OmniglotNShot(
            self.data_dir, batchsz=tasks_num,
            n_way=n_way, k_shot=k_shot,
            k_query=k_query, imgsz=imgsz)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)
