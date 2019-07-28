"""Tensorboard logger"""
from utils import Singleton
from logger import TensorboardWriter
from .base import BaseLogger


class TBLogger(BaseLogger):
    def __init__(self, config, logger):
        super().__init__()
        self.logger = logger

        self.writer = TensorboardWriter(
            config.log_dir,
            logger,
            config.trainer.tensorboard)

    def _configure(self, config):
        self.tb_config = config['trainer']['tensorboard_logs']
        self.log_index_batches = self.tb_config['index_batches']
        self.log_params = self.tb_config['log_params']
        self.log_train_images = self.tb_config['log_train_images']
        self.log_test_images = self.tb_config['log_test_images']

    def log_batch(self):
        pass

    def log_custom_metrics(self):
        pass
