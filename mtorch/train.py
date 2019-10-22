import argparse
import collections
import os
import torch
import core.data.data_loaders as module_data
import core.model.loss as module_loss
import core.model.metrics as module_metric
import core.model.arch as module_arch
import core.train.coach as module_train

from core.utils.parse_config import ConfigParser

from sacred import Experiment
from sacred.observers import MongoObserver
from core.experiment.sacred import Sacred
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
    options = []
    config = ConfigParser(args, options)

    runner = Runner(config)
    runner.run_experiment()
