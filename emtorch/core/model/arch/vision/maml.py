import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMAML(nn.Module):
    """ This is the architecture from Vinyals et al. """

    def __init__(self, inpt_c, nr_filters=64, stride=1, num_classes=10):
        super(BaseMAML, self).__init__()

        self.conv1 = nn.Conv2d(
            inpt_c, nr_filters, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nr_filters)

        self.conv2 = nn.Conv2d(
            nr_filters, nr_filters, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nr_filters)

        self.conv3 = nn.Conv2d(
            nr_filters, nr_filters, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nr_filters)

        self.conv4 = nn.Conv2d(
            nr_filters, nr_filters, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(nr_filters)
        # TODO: change input size
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=2)

        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, kernel_size=2)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out, kernel_size=2)

        out = F.relu(self.bn4(self.conv4(out)))
        out = F.max_pool2d(out, kernel_size=2)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class OmniNet(nn.Module):
    def __init__(self, inpt_c, nr_filters=32, stride=2, num_classes=10):
        super(OmniNet, self).__init__()

        self.conv1 = nn.Conv2d(
            inpt_c, nr_filters, kernel_size=3, stride=stride)
        self.conv2 = nn.Conv2d(nr_filters, nr_filters,
                               kernel_size=3, stride=stride)
        self.conv3 = nn.Conv2d(nr_filters, nr_filters,
                               kernel_size=3, stride=stride)
        self.conv4 = nn.Conv2d(nr_filters, nr_filters,
                               kernel_size=2, stride=stride)
        # TODO: change input size
        self.linear = nn.Linear(32, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class LinearNet(nn.Module):
    def __init__(self, inpt_size, num_classes=10):
        super(LinearNet, self).__init__()

        self.linear1 = nn.Linear(inpt_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 64)

        self.logits = nn.Linear(63, num_classes)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(x))
        out = F.relu(self.linear3(x))
        out = F.relu(self.linear4(x))

        out = self.logits(out)

        return out
