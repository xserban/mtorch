"""Neptune www.neptune.ml Logger"""
import torch
import os
import neptune

from .base import BaseLogger
from core.logger.neptune_helper import set_all_env


class NeptuneLogger(BaseLogger):
    def __init__(self, config):
        print("[INFO][LOGS] \t Initializing Neptune Logger ...")
        super().__init__()
        self.config = config
        self._configure(self.config)

    def _configure(self, config):
        self.l_config = config["logging"]
        self.nt_config = config["logging"]["neptune_logs"]
        try:
            set_all_env(self.nt_config["args"]["settings_file"])
            neptune.init(
                self.nt_config["args"]["user_space"] + "/" +
                self.config["project_name"])
            self.exp = neptune.create_experiment(
                name=self.config["run_name"],
                params=self.config,
                hostname=self.config["host"]["name"]
            )

            self.log_index_batches = self.l_config["index_batches"]
            self.log_params = self.l_config["log_params"]
            self.log_train_images = self.l_config["log_train_images"]
            self.log_test_images = self.l_config["log_test_images"]
        except Exception as excpt:
            print("[ERROR] \t Failed to initialize Neptune"
                  " Logger {}".format(excpt))
            raise excpt

    def _log_info(self, step, env, loss, custom_metrics, images=None):
        self.exp.log_metric(env + "." + "loss", loss)
        self.log_custom_metrics(step, env, custom_metrics)
        if self.log_train_images and images is not None:
            pass

    def log_batch(self, step, env, loss, custom_metrics, images=None):
        if self.log_index_batches:
            self._log_info(step, env, loss, custom_metrics, images)

    def log_custom_metrics(self, step, env, metrics):
        for key, value in metrics.items():
            name = env + "." + key
            self.exp.log_metric(name, value)

    def log_learning_rates(self, lrates, step):
        for index, rate in enumerate(lrates):
            name = "learning_rate" + str(index)
            self.exp.log_metric(name, rate)

    def log_epoch(self, step, env, loss, custom_metrics, lrates):
        if not self.log_index_batches:
            if lrates is not None:
                self.log_learning_rates(lrates, step)
            self._log_info(step, env, loss, custom_metrics)

    def log_parameters(self, step, env, params):
        pass

    def add_artifact(self, filename):
        try:
            self.exp.log_artifact(os.path.join(filename))
            print("[LOGGER] \t Saved artifact to Neptune database {}"
                  .format(str(filename)))
        except Exception as e:
            print("[ERROR][LOGGER] \t Could not save "
                  "artifact {} \t {}".format(filename, e))

    def stop(self):
        self.exp.stop()
