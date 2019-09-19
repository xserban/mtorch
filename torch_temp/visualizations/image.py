import matplotlib.pyplot as plt


def show_image(tensor_image, initiate_plt=True):
    if initiate_plt is True:
        plt.figure()
        plt.imshow(tensor_image.permute(1, 2, 0))
        plt.show()
    else:
        plt.imshow(tensor_image.permute(1, 2, 0))
