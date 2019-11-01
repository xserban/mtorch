from copy import deepcopy
import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from core.logger import Logger
from core.utils import read_json, write_json


class ConfigParser:
    def __init__(self, args, options="", timestamp=True):
        args = args.parse_args()
        args = vars(args)
        for key, value in options.items():
            args[key] = value

        if args["resume"]:
            self.resume = Path(args["resume"])
            self.cfg_fname = self.resume.parent / "config.json"
        else:
            msg_no_cfg = "Configuration file need to be "
            "specified. Add '-c config.json', for example."
            assert args["config"] is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = Path(args["config"])

        self._config = read_json(self.cfg_fname)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["training"]["save_dir"])
        timestamp = datetime.now().strftime(r"%m%d_%H%M%S") \
            if timestamp else ""

        exper_name = self.config["project_name"]
        self._save_dir = save_dir / "models" / exper_name / timestamp
        self._log_dir = save_dir / "log" / exper_name / timestamp

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / "config.json")

    def initialize(self, module, module_config, *args, **kwargs):
        """
        finds a function handle with the name given
        as "type" in config, and returns the
        instance initialized with corresponding
        keyword args given as "args".
        """
        module_name = module_config["type"]
        module_args = dict(module_config["args"])
        assert all([k not in module_args for k in kwargs]
                   ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def init_logger(self, sacred_ex=None):
        # configure logging module
        self.logger = Logger(self.config, self.log_dir,
                             {
                                 0: logging.WARNING,
                                 1: logging.INFO,
                                 2: logging.DEBUG,
                                 3: logging.ERROR
                             },
                             sacred_ex=sacred_ex)

    def get_logger(self, name, verbosity=2):
        if not self.logger:
            raise("Please initialize logger")

        return self.logger.get_py_logger(name, verbosity)

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
