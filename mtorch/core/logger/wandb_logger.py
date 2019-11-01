"""Weights and Biases www.wandb.com Logger"""
from .base import BaseLogger
import wandb


class WANDBLogger(BaseLogger):
    def __init__(self, config, pylogger):
        print("[INFO][LOGS] \t Initializing Weights and Biases Logger ...")
        super().__init__()
        self.config = config
        self.pylogger = pylogger
        self._configure(self.config)

    def _configure(self, config):
        self.l_config = config["logging"]
        self.wb_config = config["logging"]["wandb_logs"]
        try:
            wandb.init(name=self.config["run_name"],
                       project=self.config["project_name"],
                       config=self.config.config)
            self.log_index_batches = self.l_config["index_batches"]
            self.log_params = self.l_config["log_params"]
            self.log_train_images = self.l_config["log_train_images"]
            self.log_test_images = self.l_config["log_test_images"]
        except Exception as e:
            self.pylogger.error("Failed to initialize W&B Logger {}".format(e))

    def _log_info(self, step, env, loss, custom_metrics, images=None):
        wandb.log({env + "." + "loss": loss}, step=step)
        self.log_custom_metrics(step, env, custom_metrics)
        if self.log_train_images and images is not None:
            for img in images:
                wandb.log({env + ".input": [wandb.Image(img)]})

    def log_batch(self, step, env, loss, custom_metrics, images=None):
        if self.log_index_batches:
            self._log_info(step, env, loss, custom_metrics, images)

    def log_custom_metrics(self, step, env, metrics):
        base_name = env
        for key, value in metrics.items():
            name = base_name + "." + key
            wandb.log({name, value}, step=step)

    def log_learning_rates(self, lrates, step):
        for index, rate in enumerate(lrates):
            name = "learning_rate" + str(index)
            wandb.log({name: rate}, step=step)

    def log_epoch(self, step, env, loss, custom_metrics, lrates):
        if not self.log_index_batches:
            name = env + "." + "loss"
            wandb.log({name: loss})
            if lrates is not None:
                self.log_learning_rates(lrates, step)
            self._log_info(step, env, loss, custom_metrics)

    def log_parameters(self, step, env, params):
        if self.log_params:
            for name, param in params:
                wandb.log(name, wandb.Histogram(param), step=step)

    def add_artifact(self, filename):
        try:
            wandb.save(filename)
            print("[LOGGER] \t Saved artifact to WANDB database {}"
                  .format(str(filename)))
        except Exception as e:
            print("[ERROR][LOGGER] \t Could not save "
                  "artifact {} \t {}".format(filename, e))
