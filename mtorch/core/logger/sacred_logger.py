"""Sacred Logger"""
from .base import BaseLogger


class SacredLogger(BaseLogger):
    def __init__(self, config, sacred_ex):
        print("[INFO][LOGS] \t Initializing Sacred Logger ...")
        super().__init__()
        self._configure(config)
        self.sacred_ex = sacred_ex

    def _configure(self, config):
        self.tb_config = config["logging"]
        self.log_index_batches = self.tb_config["index_batches"]
        self.log_params = self.tb_config["log_params"]
        self.log_train_images = self.tb_config["log_train_images"]
        self.log_test_images = self.tb_config["log_test_images"]

    def log_batch(self, step, env, loss, custom_metrics):
        if self.log_index_batches:
            name = env + "." + "loss"  # + "." + str(step)
            self.sacred_ex.log_scalar(name, loss)
            self.log_custom_metrics(step, env, custom_metrics)

    def log_custom_metrics(self, step, env, metrics):
        base_name = env  # + "." + str(step)
        for key, value in metrics.items():
            name = base_name + "." + key
            self.sacred_ex.log_scalar(name, value)

    def log_learning_rates(self, lrates):
        for index, rate in enumerate(lrates):
            name = "learning_rate_" + str(index)
            self.sacred_ex.log_scalar(name, rate)

    def log_epoch(self, step, env, loss, custom_metrics, lrates):
        if not self.log_index_batches:
            name = env + "." + "loss"  # + "." + str(step)
            self.sacred_ex.log_scalar(name, loss)
            if lrates is not None:
                self.log_learning_rates(lrates)
            self.log_custom_metrics(step, env, custom_metrics)

    def log_parameters(self, step, env, params):
        pass

    def add_artifact(self, filename, name, metadata=None):
        self.sacred_ex.add_artifact(filename, name, metadata)
