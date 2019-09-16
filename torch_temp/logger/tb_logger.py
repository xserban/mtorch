"""Tensorboard logger"""
from torch_temp.logger.tensorboard_writer import TensorboardWriter
from torchvision.utils import make_grid
from .base import BaseLogger


class TBLogger(BaseLogger):
    def __init__(self, log_dir, config):
        print("[INFO][LOGS] \t Initializing Tensorboard Logger ...")
        super().__init__()

        self.log_dir = log_dir
        self.writer = TensorboardWriter(self.log_dir)

        self._configure(config)

    def _configure(self, config):
        self.tb_config = config["logging"]
        self.log_index_batches = self.tb_config["index_batches"]
        self.log_params = self.tb_config["log_params"]
        self.log_train_images = self.tb_config["log_train_images"]
        self.log_test_images = self.tb_config["log_test_images"]

    def log_batch(self, step, env, loss, custom_metrics, images=None):
        if self.log_index_batches:
            self.writer.set_step(step, env)
            self.writer.add_scalar("loss", loss)
            self.log_custom_metrics(custom_metrics)

            if self.log_train_images and images is not None:
                self.writer.add_image("input",
                                      make_grid(images.cpu(), nrow=8,
                                                normalize=True))

    def log_custom_metrics(self, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar("{}".format(key), value)

    def log_learning_rates(self, lrates):
        for index, rate in enumerate(lrates):
            name = "learning_rate_" + str(index)
            self.writer.add_scalar("{}".format(name), rate)

    def log_epoch(self, step, env, loss, lrates, custom_metrics):
        if not self.log_index_batches:
            self.writer.set_step(step, env)
            self.log_learning_rates(lrates)
            self.log_batch(step, env, loss, custom_metrics)

    def log_parameters(self, step, env, params):
        if self.log_params:
            self.writer.set_step(step, env)
            for name, param in params:
                self.writer.add_histogram(name, param, bins="auto")
