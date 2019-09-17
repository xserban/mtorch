###
# Adapted from:
# https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py
###


import os
import os.path
import errno
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from torch_temp.utils.download_data import download


class OmniglotDataset(data.Dataset):
    urls = [
        "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
        "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"
    ]
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "training.pt"
    test_file = "test.pt"

    """
    The items are (filename,category). The index of all the categories
    can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    """

    def __init__(self, root, transform=None,
                 target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                download()
            else:
                raise RuntimeError("Dataset not found." +
                                   " You can use download=True to download it")

        self.all_items = find_classes(
            os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join("/", [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           "images_evaluation")) \
            and os.path.exists(os.path.join(
                self.root, self.processed_folder, "images_background"))


def find_classes(root_dir):
    retour = []
    for (root, _, files) in os.walk(root_dir):
        for file in files:
            if file.endswith("png"):
                retr = root.split("/")
                lngrtr = len(retr)
                retour.append(
                    (file, retr[lngrtr - 2] + "/" + retr[lngrtr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


class OmniglotNShot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        if not os.path.isfile(os.path.join(root, "omniglot.npy")):
            # if root/data.npy does not exist, just download it
            self.input = OmniglotDataset(root, download=True,
                                         transform=transforms.Compose([
                                             lambda x: Image.open(
                                                 x).convert("L"),
                                             lambda x: x.resize(
                                                 (imgsz, imgsz)),
                                             lambda x: np.reshape(
                                                 x, (imgsz, imgsz, 1)),
                                             lambda x: np.transpose(
                                                 x, [2, 0, 1]),
                                             lambda x: x/255.])
                                         )

            # {label:img1, img2..., 20 imgs, label2: img1,
            # img2,... in total, 1623 label}
            temp = dict()
            for (img, label) in self.input:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.input = []
            # labels info deserted , each label contains 20imgs
            for label, imgs in temp.items():
                self.input.append(np.array(imgs))

            # as different class may have different number of imgs
            # [[20 imgs],..., 1623 classes in total]
            self.input = np.array(self.input).astype(np.float)
            # each character contains 20 imgs
            print("data shape:", self.input.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, "omniglot.npy"), self.input)
            print("write into omniglot.npy.")
        else:
            # if data.npy exists, just load it.
            self.input = np.load(os.path.join(root, "omniglot.npy"))
            print("load from omniglot.npy.")

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep
        # training and test set distinct!
        self.input_train, self.input_test = self.input[:1200], self.input[1200:]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.input.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <= 20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.input_train,
                         "test": self.input_test}  # original data cached
        print("DB: train", self.input_train.shape,
              "test", self.input_test.shape)

        self.datasets_cache = {"train":
                               self.load_data_cache(self.datasets["train"]),
                               "test":
                               self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.input_train)
        self.std = np.std(self.input_train)
        self.max = np.max(self.input_train)
        self.min = np.min(self.input_train)
        # print("before norm:", "mean", self.mean, "max",
        # self.max, "min", self.min, "std", self.std)
        self.input_train = (self.input_train - self.mean) / self.std
        self.input_test = (self.input_test - self.mean) / self.std

        self.mean = np.mean(self.input_train)
        self.std = np.std(self.input_train)
        self.max = np.max(self.input_train)
        self.min = np.min(self.input_train)

    # print("after norm:", "mean", self.mean, "max",
    # self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y,
          target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print("preload next 50 caches of batchsz of batch.")
        for _ in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for _ in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(
                    data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(
                        20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class]
                                 [selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class]
                                 [selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(
                    self.n_way * self.k_shot, 1,
                    self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(
                    self.n_way * self.k_query, 1,
                    self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(
                    self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(
                self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(
                np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(
                self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(
                np.int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode="train"):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(
                self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
