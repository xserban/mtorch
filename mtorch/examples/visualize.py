import matplotlib.pyplot as plt

from core import visualizations as viz
from core.utils import util
from core.data.transformations import *
from core.data.data_loaders.vision import *


def main():
    training_dl_cifar = CIFAR10Loader(
        "downloaded-data/",
        batch_size=None,
        training=True,
        transformations="CIFARTransformations"
    )

    training_dl_default = CIFAR10Loader(
        "downloaded-data/",
        batch_size=None,
        training=True,
        transformations="DefaultTransformations"
    )

    testing_dl_cifar = CIFAR10Loader(
        "downloaded-data/",
        batch_size=None,
        training=False,
        transformations="CIFARTransformations"
    )

    testing_dl_default = CIFAR10Loader(
        "downloaded-data/",
        batch_size=None,
        training=False,
        transformations="DefaultTransformations"
    )

    images, labels, _ = util.sample_n_datapoints(training_dl_default.dataset.data,
                                                 training_dl_default.dataset.targets,
                                                 1)
