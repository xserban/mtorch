import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv(nn.Module):
    def __init__(self, in_size, *args, **kwargs):
        super(SimpleConv, self).__init__()
