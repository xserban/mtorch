"""This class controlls all loggers.
Currently, there are 3 loggers implemented:
  - pylogger - logging module from python
  - tb_logger - tensorboard logger
  - sacred_logger - sacred
"""
import logging
import logging.config
from pathlib import Path
from utils import Singleton
from utils import read_json
from logger.tb_logger import TBLogger
from logger.sacred_logger import SacredLogger
from logger.base import BaseLogger


class Logger(BaseLogger, metaclass=Singleton):
    def __init__(self,
                 config,
                 log_dir,
                 log_levels,
                 py_log_config='configs/py_logger_config.json',
                 py_default_level=logging.INFO,
                 sacred_ex=None):
        super(Logger, self).__init__()

        self.log_dir = log_dir
        self.log_levels = log_levels

        self.init_py_logger(self.log_dir, py_log_config, py_default_level)
        self.init_tb_logger(config)
        self.init_sacred_logger(config, sacred_ex)

    def init_py_logger(self, save_dir, log_config, default_level=logging.INFO):
        log_config = Path(log_config)
        if log_config.is_file():
            config = read_json(log_config)
            # modify logging paths based on run config
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(save_dir / handler['filename'])

            logging.config.dictConfig(config)
        else:
            print(
                "Warning: logging configuration file is not found in {}.".format(log_config))
            logging.basicConfig(level=default_level)

    def init_tb_logger(self, config):
        if config['logger']['tensorboard_logs']['do'] is True:
            self.tb_logger = TBLogger(self.log_dir, config)
        else:
            self.tb_logger = None

    def init_sacred_logger(self, config, sacred_ex):
        if config['logger']['sacred_logs']['do'] is True:
            self.sacred_logger = SacredLogger(config, sacred_ex)
        else:
            self.sacred_logger = None

    def get_py_logger(self, name, verbosity):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def log_batch(self, step, env, loss, custom_metrics, images):
        if self.tb_logger is not None:
            self.tb_logger.log_batch(step, env, loss, custom_metrics, images)
        if self.sacred_logger is not None:
            self.sacred_logger.log_batch(step, env, loss, custom_metrics)

    def log_epoch(self, step, env, loss, custom_metrics):
        if self.tb_logger is not None:
            self.tb_logger.log_epoch(step, env, loss, custom_metrics)
        if self.sacred_logger is not None:
            self.sacred_logger.log_epoch(step, env, loss, custom_metrics)

    def log_validation_params(self, step, env, parameters):
        if self.tb_logger is not None:
            self.tb_logger.log_parameters(step, env, parameters)
