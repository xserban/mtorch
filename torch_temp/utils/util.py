import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict


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


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
