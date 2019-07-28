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


class Logger(metaclass=Singleton):
    def __init__(self,
                 config,
                 log_dir,
                 log_levels,
                 py_log_config='configs/py_logger_config.json',
                 py_default_level=logging.INFO):
        super(Logger, self).__init__()

        self.log_dir = log_dir
        self.log_levels = log_levels

        self._init_py_logger(self.log_dir, py_log_config, py_default_level)

    def init_all_loggers(self):
        pass

    def _init_py_logger(self, save_dir, log_config, default_level=logging.INFO):
        log_config = Path(log_config)
        if log_config.is_file():
            config = read_json(log_config)
            print(config)
            # modify logging paths based on run config
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(save_dir / handler['filename'])

            logging.config.dictConfig(config)
        else:
            print(
                "Warning: logging configuration file is not found in {}.".format(log_config))
            logging.basicConfig(level=default_level)

    def _init_tb_logger(self):
        pass

    def _init_sacred_logger(self):
        pass

    def get_py_logger(self, name, verbosity):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
