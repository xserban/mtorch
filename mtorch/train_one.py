import argparse
import collections

from core.utils.parse_config import ConfigParser
from core.experiment.runner import Runner


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Experiment Management")
    args.add_argument("-c", "--config", default=None, type=str,
                      help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="indices of GPUs to enable (default: all)")

    # custom cli options to modify configuration
    # from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = {}
    config = ConfigParser(args, options)

    runner = Runner(config)
    runner.run_experiment()
