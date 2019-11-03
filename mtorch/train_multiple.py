import argparse

from core.experiment.mrunners import MRunners
from core.utils.parse_config import ConfigParser
from core.utils.util import read_dir_files, read_json, set_seed

from pathlib import Path

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Experiment Management")
    args.add_argument("-f", "--folder", default=None, type=str,
                      help="configs fodler path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to latest checkpoint (default: None)")

    fargs = args.parse_args()
    assert fargs.folder is not None, "A folder with config files must be " \
        "specified"

    fpath = Path(fargs.folder)
    files = read_dir_files(fargs.folder)

    configs = []
    for f in files:
        configs.append(ConfigParser(args, options={"config": str(fpath / f)}))

    set_seed(0)
    runner = MRunners(configs)
