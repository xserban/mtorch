import os
from pathlib import Path
from core.utils import read_json


def set_all_env(file_name):
    js = read_json(Path(file_name), dic=False)
    for key, value in js.items():
        os.environ[key] = value


if __name__ == "__main__":
    set_all_env()
