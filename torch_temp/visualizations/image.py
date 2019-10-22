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


def imshow(images):
    images = torchvision.utils.make_grid(images)
    images = images / 2 + 0.5  # unnormalize
    npimg = images.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_image_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

    plt.figure()
    plt.show()
