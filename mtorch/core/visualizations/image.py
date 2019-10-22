import matplotlib.pyplot as plt
import numpy as np
import torchvision


def show_image(tensor_image, initiate_plt=True):
    if initiate_plt is True:
        plt.figure()
        plt.imshow(tensor_image.permute(1, 2, 0))
        plt.show()
    else:
        plt.imshow(tensor_image.permute(1, 2, 0))


def convert_to_imshow_format(image):
    image = image.numpy()
    # convert from CHW to HWC
    # e.g. from 3x32x32 to 32x32x3
    return image.transpose(1, 2, 0)


def imshow(images, file=None):
    """Displays images in a grid. This
      method expects the images to be unnormalized
    :param images: array or tensor of images
    """
    images = [convert_to_imshow_format(x) for x in images]
    images = torchvision.utils.make_grid(images)
    npimg = images.numpy()
    plt.imshow(npimg)
    plt.show()

    if file is not None:
        plt.savefig(file)


def show_image_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

    plt.figure()
    plt.show()
