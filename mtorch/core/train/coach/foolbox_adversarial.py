import advertorch.attacks as attacks
from advertorch.context import ctx_noparamgrad_and_eval

import numpy as np
import torch

from tqdm import tqdm
from core.train.coach.base import BaseTrainer
from core.utils import inf_loop


class FoolboxAdversarialTrainer(BaseTrainer):
    """Performs adversarial training
       using a given attack
    """

    def __init__(self, model, loss, metrics, optimizer, config,
                 train_data_loader,
                 attack_type,
                 attack_params,
                 worst_case_training,
                 valid_data_loader=None,
                 test_data_loader=None,
                 len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)

        self.config = config
        self.train_data_loader = train_data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.do_testing = self.test_data_loader is not None

        self.lrates = self.get_lrates()

        # init adversarial attack
        if attack_params["ratio"] is not None:
            attack_params["kwargs"]["eps"] /= attack_params["ratio"]
            attack_params["kwargs"]["eps_iter"] /= attack_params["ratio"]
        self._init_attack(attack_type, attack_params["kwargs"])
        self.worst_case_training = worst_case_training

    def _init_attack(self, attack_type, attack_parameters):
        """Initializes adversarial attack
        :param attack_type: attack name from advertorch
        :param attack_parameters: dictionary with parameters for the attack
        """
        try:
            self.attack_type = attack_type
            self.attack_parameters = attack_parameters
            self.adversary = getattr(attacks, attack_type)(
                self.model,
                # loss_fn=self.loss,
                loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                **attack_parameters)
        except Exception as e:
            raise e

    def _train_epoch(self, epoch):
        """Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information to be saved.
        Note:
            For additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key "metrics".
        """
        print("[INFO][TRAIN] \t Starting Training Epoch {}:".format(epoch))
        self.model.train()
        total_loss, total_adversarial_loss, total_metrics = 0, 0, np.zeros(
            len(self.metrics))

        for batch_idx, (data, target) in \
                enumerate(tqdm(self.train_data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            # run batch and get loss
            loss, adv_loss, metrics, _ = self._run_batch(
                data, target, eval_metrics=True)
            total_loss += loss
            total_adversarial_loss += adv_loss
            total_metrics += metrics
            # log info specific to this batch
            self.logger.log_batch((epoch - 1) * self.len_epoch + batch_idx,
                                  "train",
                                  loss,
                                  metrics,
                                  data)

            if batch_idx == self.len_epoch:
                break
        # log info specific to the whole epoch
        avg_loss = total_loss / self.len_epoch
        avg_loss_adv = total_adversarial_loss / self.len_epoch
        avg_metrics = (total_metrics /
                       len(self.train_data_loader)).tolist()
        # add adversarial loss to custom metrics
        metr = self.get_metrics_dic(avg_metrics)
        metr["adversarial_loss"] = avg_loss_adv
        self.logger.log_epoch(epoch - 1, "train",
                              avg_loss,
                              metr,
                              self.lrates)
        log = {
            "loss": avg_loss,
            "adversarial_loss": avg_loss_adv,
            "train_metrics": avg_metrics
        }
        # run validation and testing
        self._validate_and_test(epoch, log)
        self.adapt_lr(epoch)

        return log

    ###
    # Epoch helpers
    ###
    def _run_batch(self, data, target, eval_metrics=False,
                   train=True):
        """Runs batch optimization and returns loss
        :param data: input batch
        :param target: labels batch
        :return: loss value
        """
        # generate adversarial data
        original_data = data
        with ctx_noparamgrad_and_eval(self.model):
            adversarial_data = self.adversary.perturb(data, target)
        self.optimizer.zero_grad()
        # get adversarial output and normal+adversarial loss
        if train is True:
            output = self.model(original_data)
            adv_output = self.model(adversarial_data)
        else:
            with torch.no_grad():
                output = self.model(original_data)
                adv_output = self.model(adversarial_data)

        adv_loss = self.loss(adv_output, target)
        loss = self.loss(output, target)

        if train is True:
            if self.worst_case_training is False:
                loss.backward()
                self.optimizer.step()
            adv_loss.backward()
            self.optimizer.step()

        if eval_metrics is True:
            metrics = self.eval_metrics(output, adv_output, target)
            return loss.item(), adv_loss.item(), metrics, \
                self.get_metrics_dic(metrics)
        else:
            return loss.item(), adv_loss.item()

    def _validate_and_test(self, epoch, log):
        """Run validation and testing"""
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.do_testing and epoch % self.test_epochs_interval == 0:
            test_log = self._test_epoch(epoch)
            log.update(test_log)

    def _valid_epoch(self, epoch):
        """Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key "val_metrics".
        """
        print("[INFO][VALIDATION] \t "
              "Starting Validation Epoch {}:".format(epoch))
        self.model.eval()
        total_val_loss, total_adv_loss, total_val_metrics = 0, 0, np.zeros(
            len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            # get loss and run metrics
            loss, adv_loss, metrics, dic_metrics = self._run_batch(
                data, target, eval_metrics=True, train=False)
            total_val_loss += loss
            total_adv_loss += adv_loss
            total_val_metrics += metrics
            # log results specific to batch
            self.logger.log_batch((epoch - 1) *
                                  len(self.valid_data_loader) +
                                  batch_idx,
                                  "valid",
                                  loss,
                                  dic_metrics,
                                  data)
        # log info specific to the whole validation epoch
        avg_loss = total_val_loss / len(self.valid_data_loader)
        avg_loss_adv = total_adv_loss / len(self.valid_data_loader)
        avg_metrics = (total_val_metrics /
                       len(self.valid_data_loader)).tolist()
        # add adversarial loss to custom metrics
        metr = self.get_metrics_dic(avg_metrics)
        metr["adversarial_loss"] = avg_loss_adv
        self.logger.log_epoch(epoch - 1, "valid",
                              avg_loss,
                              metr,
                              None)
        # add histogram of model parameters to the tensorboard
        self.logger.log_validation_params(
            epoch-1, "valid", self.model.named_parameters())
        # return final log metrics
        return {
            "val_loss": avg_loss,
            "val_adv_loss": avg_loss_adv,
            "val_metrics": avg_metrics
        }

    def _test_epoch(self, epoch):
        print("[INFO][TEST] \t Starting Test Epoch {}:".format(epoch))
        self.model.eval()
        total_test_loss, total_adv_loss, total_test_metrics = 0, 0, np.zeros(
            len(self.metrics))

        for i, (data, target) in enumerate(tqdm(self.test_data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            # get loss and and run metrics
            loss, adv_loss, metrics, dic_metrics = self._run_batch(
                data, target, eval_metrics=True, train=False)
            total_test_loss += loss
            total_adv_loss += adv_loss
            total_test_metrics += metrics
            # log results specific to batch
            self.logger.log_batch((epoch - 1) *
                                  len(self.test_data_loader) + i,
                                  "test",
                                  loss,
                                  dic_metrics,
                                  data)
        # log results specific to epoch
        avg_loss = total_test_loss / len(self.test_data_loader)
        avg_loss_adv = total_adv_loss / len(self.test_data_loader)
        avg_metrics = (total_test_metrics /
                       len(self.test_data_loader)).tolist()
        # add adversarial loss to custom metrics
        metr = self.get_metrics_dic(avg_metrics)
        metr["adversarial_loss"] = avg_loss_adv
        self.logger.log_epoch(epoch - 1, "test",
                              avg_loss,
                              metr,
                              None)
        # return final log metrics
        return {
            "test_loss": avg_loss,
            "test_adv_loss": avg_loss_adv,
            "test_metrics": avg_metrics
        }

    def eval_metrics(self, output, adversarial_output, target):
        """Evaluate all metrics"""
        metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            if hasattr(metric, "adversarial") and metric.adversarial is True:
                metrics[i] = metric.forward(adversarial_output, target)
            else:
                metrics[i] = metric.forward(output, target)
        return metrics
