"""
  This module runs an experiment from a configi file
"""
import argparse
import collections
import os
import torch

import core.data.data_loaders as module_data
import core.model.loss as module_loss
import core.model.metrics as module_metric
import core.model.arch as module_arch
import core.train.coach as module_train

from core.utils.parse_config import ConfigParser

from sacred import Experiment
from sacred.observers import MongoObserver
from core.experiment.sacred import Sacred

# set multiprocessing
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Runner:
    def __init__(self, config):
        self.config = config

    def run_experiment(self):
        if self.config["logging"]["sacred_logs"]["do"] is False:
            self.config.init_logger()
            self.run()
        else:
            from .sacred_wrapper import ex
            sacred_exp = Sacred(
                ex,
                config=self.config.config,
                auto_config=True,
            )
            self.config.init_logger(sacred_ex=sacred_exp.ex)
            ex.main_ = self.run
            ex.run(options={'--name': self.config["project_name"]})

    def run(self):
        self.logger = self.config.get_logger("train")

        train_data_loader, valid_data_loader, test_data_loader = \
            self.init_data_loaders()
        model, loss, metrics = self.init_model()
        optimizer = self.init_optimizer(model)

        trainer_args = {
            "model": model,
            "loss": loss,
            "metrics": metrics,
            "optimizer": optimizer,
            "config": self.config,
            "train_data_loader": train_data_loader,
            "valid_data_loader": valid_data_loader,
            "test_data_loader": test_data_loader,
        }

        trainer = self.config.initialize(
            module_train, self.config["training"]["trainer"], **trainer_args)
        trainer.train()

    def init_data_loaders(self):
        train_data_loader = self.config.initialize(
            module_data, self.config["data"]["loader"],
            **self.config["data"]["loader"]["kwargs"])
        valid_data_loader = train_data_loader.split_validation()

        if self.config["testing"]["do"]:
            # TODO: find a better manner to write this call
            test_data_loader = getattr(module_data,
                                       self.config["data"]["loader"]["type"])(
                self.config["data"]["loader"]["args"]["data_dir"],
                batch_size=self.config["testing"]["test_batch_size"],
                shuffle=False,
                validation_split=0.0,
                training=False,
                transformations=self.config["data"]["loader"]["args"]["transformations"],
                **self.config["data"]["loader"]["kwargs"]
            )
        else:
            test_data_loader = None

        return train_data_loader, valid_data_loader, test_data_loader

    def init_model(self):
        model = self.config.initialize(
            module_arch, self.config["model"]["arch"])
        self.logger.info(model)
        loss = self.config.initialize(
            module_loss, self.config["model"]["loss_function"])
        metrics = [self.config.initialize(module_metric, met)
                   for met in self.config["metrics"]]
        return model, loss, metrics

    def init_optimizer(self, model):
        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters())
        return self.config.initialize(
            torch.optim, self.config["optimizer"]["opt"], trainable_params)
