"""This class controlls all loggers.
Currently, there are 3 loggers implemented:
  - pylogger - logging module from python
  - tb_logger - tensorboard logger
  - sacred_logger - sacred
"""
import logging
import logging.config
from pathlib import Path
from core.utils import Singleton
from core.utils import read_json
from core.logger.tb_logger import TBLogger
from core.logger.sacred_logger import SacredLogger
from core.logger.wandb_logger import WANDBLogger
from core.logger.neptune_logger import NeptuneLogger
from core.logger.elasticinfra_logger import InfraLogger
from core.logger.base import BaseLogger


class Logger(BaseLogger, metaclass=Singleton):
    def __init__(self,
                 config,
                 log_dir,
                 log_levels,
                 py_log_config="mtorch/core/logger/py_logger_config.json",
                 py_default_level=logging.ERROR,
                 sacred_ex=None):
        super(Logger, self).__init__()
        self.log_dir = log_dir
        self.log_levels = log_levels
        self.default_log_level = py_default_level

        self.init_py_logger(self.log_dir, py_log_config,
                            self.default_log_level)
        self.init_tb_logger(config)
        self.init_sacred_logger(config, sacred_ex)
        self.init_wandb_logger(config)
        self.init_neptune_logger(config)
        self.init_infrastructure_logger(config)

    def init_py_logger(self, save_dir,
                       log_config,
                       default_level=logging.ERROR):
        log_config = Path(log_config)
        if log_config.is_file():
            config = read_json(log_config)
            # modify logging paths based on run config
            for _, handler in config["handlers"].items():
                if "filename" in handler:
                    handler["filename"] = str(save_dir / handler["filename"])

            logging.config.dictConfig(config)
        else:
            print("Warning: logging configuration "
                  "file is not found in {}.".format(log_config))
            logging.basicConfig(level=default_level)

    def init_tb_logger(self, config):
        if config["logging"]["tensorboard_logs"]["do"] is True:
            self.tb_logger = TBLogger(self.log_dir, config)
        else:
            self.tb_logger = None

    def init_sacred_logger(self, config, sacred_ex):
        if config["logging"]["sacred_logs"]["do"] is True:
            self.sacred_logger = SacredLogger(config, sacred_ex)
        else:
            self.sacred_logger = None

    def init_wandb_logger(self, config):
        if config["logging"]["wandb_logs"]["do"] is True:
            self.wandb_logger = WANDBLogger(config)
        else:
            self.wandb_logger = None

    def init_neptune_logger(self, config):
        if config["logging"]["neptune_logs"]["do"] is True:
            self.neptune_logger = NeptuneLogger(config)
        else:
            self.neptune_logger = None

    def init_infrastructure_logger(self, config):
        elk_logger = self.get_py_logger("elk_logger", 3)
        if config["logging"]["infrastructure_logs"]["do"] is True:
            self.infra_logger = InfraLogger(
                config,
                elk_logger)
            # configure logger from the elasticsearch module
            es_logger = logging.getLogger("elasticsearch")
            es_logger.setLevel(self.default_log_level)
        else:
            self.infra_logger = None

    def get_py_logger(self, name, verbosity=2):
        msg_verbosity = \
            "verbosity option {} is invalid. Valid options are {}.".format(
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
        if self.wandb_logger is not None:
            self.wandb_logger.log_batch(
                step, env, loss, custom_metrics, images)
        if self.neptune_logger is not None:
            self.neptune_logger.log_batch(
                step, env, loss, custom_metrics, images)

    def log_epoch(self, step, env, loss, custom_metrics, lrates=None):
        if self.tb_logger is not None:
            self.tb_logger.log_epoch(step, env, loss, custom_metrics, lrates)
        if self.sacred_logger is not None:
            self.sacred_logger.log_epoch(
                step, env, loss, custom_metrics, lrates)
        if self.wandb_logger is not None:
            self.wandb_logger.log_epoch(
                step, env, loss, custom_metrics, lrates)
        if self.neptune_logger is not None:
            self.neptune_logger.log_epoch(
                step, env, loss, custom_metrics, lrates)

    def log_validation_params(self, step, env, parameters):
        if self.tb_logger is not None:
            self.tb_logger.log_parameters(step, env, parameters)
        if self.wandb_logger is not None:
            self.wandb_logger.log_parameters(step, env, parameters)
        if self.neptune_logger is not None:
            self.neptune_logger.log_parameters(step, env, parameters)

    def log_custom_metrics(self, metrics):
        super().log_custom_metrics(metrics)

    def log_artifact(self, filename, name, metadata=None):
        """Stores an artifact in the database
          The artifact is typically a model's weights
          or a special file
        """
        if self.sacred_logger is not None:
            self.sacred_logger.add_artifact(filename, name, metadata)
        if self.wandb_logger is not None:
            self.wandb_logger.add_artifact(filename)
        if self.neptune_logger is not None:
            self.neptune_logger.add_artifact(filename)

    def start_loops(self):
        """This method will start all loggers that
          run in a loop, on a separate thread"""
        if self.infra_logger is not None:
            self.infra_logger.start()

    def stop_loops(self):
        """This method will stop all loggers that
          run in a loop, on a separate thread"""
        if self.infra_logger is not None:
            self.infra_logger.stop()

    def stop_experiments(self):
        self.stop_loops()

        if self.neptune_logger is not None:
            self.neptune_logger.stop()
