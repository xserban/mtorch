"""Weights and Biases www.wandb.com Logger"""
import torch
import os
import wandb

from .base import BaseLogger


class WANDBLogger(BaseLogger):
    def __init__(self, config):
        print("[INFO][LOGS] \t Initializing Weights and Biases Logger ...")
        super().__init__()
        self.config = config
        self._configure(self.config)

    def _configure(self, config):
        self.l_config = config["logging"]
        self.wb_config = config["logging"]["wandb_logs"]
        try:
            wandb.init(name=self.config["run_name"],
                       project=self.config["project_name"],
                       config=self.config,
                       dir=self.config["training"]["save_dir"])
            self.log_index_batches = self.l_config["index_batches"]
            self.log_params = self.l_config["log_params"]
            self.log_train_images = self.l_config["log_train_images"]
            self.log_test_images = self.l_config["log_test_images"]
        except Exception as e:
            print("[ERROR] \t Failed to initialize W&B Logger {}".format(e))
            raise e

    def _log_info(self, step, env, loss, custom_metrics, images=None):
        wandb.log({env + "." + "loss": loss})
        self.log_custom_metrics(step, env, custom_metrics)
        if self.log_train_images and images is not None:
            for img in images:
                with torch.no_grad():
                    wandb.log({env + ".input": [wandb.Image(img.cpu())]})

    def log_batch(self, step, env, loss, custom_metrics, images=None):
        if self.log_index_batches:
            self._log_info(step, env, loss, custom_metrics, images)

    def log_custom_metrics(self, step, env, metrics):
        for key, value in metrics.items():
            name = env + "." + key
            wandb.log({name: value})

    def log_learning_rates(self, lrates, step):
        for index, rate in enumerate(lrates):
            name = "learning_rate" + str(index)
            wandb.log({name: rate})

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
                with torch.no_grad():
                    wandb.log({name: wandb.Histogram(
                        param.cpu())})

    def add_artifact(self, filename):
        try:
            wandb.save(os.path.join(filename))
            print("[LOGGER] \t Saved artifact to WANDB database {}"
                  .format(str(filename)))
        except Exception as e:
            print("[ERROR][LOGGER] \t Could not save "
                  "artifact {} \t {}".format(filename, e))
