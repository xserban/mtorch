import torch
import os
import shutil
from typing import Tuple, List


def mkdir(dir):
    try:
        os.mkdir(dir)
    except Exception as e:
        raise e


def rmdir(dir):
    try:
        shutil.rmtree(dir)
    except Exception as e:
        raise e
