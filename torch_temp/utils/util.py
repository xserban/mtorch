import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
import numpy as np
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname, dic=True):
    with fname.open("rt") as handle:
        if dic is True:
            return json.load(handle, object_hook=OrderedDict)
        return json.load(handle)


def write_json(content, fname):
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """Wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def sample_n_datapoints(data, labels, n=100):
    """Selects n random datapoints and their
    corresponding labels from a dataset
    """
    assert len(data) == len(labels)
    # convert to np arrays because labels
    # are usually python lists
    data = np.array(data)
    labels = np.array(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
