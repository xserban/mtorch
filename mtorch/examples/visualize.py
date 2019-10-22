from ..core.data.data_loaders.vision import *
from ..core.data.transformations import *
import core.utils.util as utl
import core.visualizations as viz

import matplotlib.pyplot as plt

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

images, labels, _ = utl.sample_n_datapoints(training_dl_default.dataset.data,
                                            training_dl_default.dataset.targets,
                                            1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
for idx, image in enumerate(images):
    axes[idx].imshow(utl.convert_to_imshow_format(image))
    axes[idx].set_title(classes[labels[idx]])
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

print('SHOWING IMAGES')

print('STOP')
