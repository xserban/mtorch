# implementation for paper https://arxiv.org/pdf/1904.12843.pdf
# currently experimental
import advertorch.attacks as attacks
from advertorch.context import ctx_noparamgrad_and_eval

import numpy as np
import torch

from tqdm import tqdm
from core.train.coach.base import BaseTrainer
from core.utils import inf_loop


class FreeAdversarialTrainer(BaseTrainer):
    """Performs free adversarial training
    and evaluates results against given attack
    """

    def __init__(self, model, loss, metrics, optimizer, config,
                 train_data_loader,
                 batch_iterations,
                 attack_type,
                 attack_params,
                 eps=4.0,
                 new_perturbation=False,
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

        # free adversarial
        self.batch_iterations = batch_iterations
        self.new_perturbation = new_perturbation
        self.eps = eps/255.0
        # TODO: crop images / load automatically nr.
        # of channels and image size
        self._init_perturbation()

        # init adversarial attack
        if attack_params["ratio"] is not None:
            attack_params["kwargs"]["eps"] /= attack_params["ratio"]
            attack_params["kwargs"]["eps_iter"] /= attack_params["ratio"]
        self._init_attack(attack_type, attack_params["kwargs"])

    def _init_perturbation(self):
        self.perturbation = torch.zeros(
            self.config["data"]["loader"]["args"]["batch_size"],
            3,
            32,
            32
        ).to(self.device)

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
                # TODO: maybe change this
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
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in \
                enumerate(tqdm(self.train_data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            # run batch and get loss
            loss = self._run_batch(
                data, target, eval_metrics=True)
            total_loss += loss
            # total_metrics += metrics
            # log info specific to this batch
            self.logger.log_batch((epoch - 1) * self.len_epoch + batch_idx,
                                  "train",
                                  loss,
                                  {},
                                  data)

            if batch_idx == self.len_epoch:
                break
        # log info specific to the whole epoch
        total_train_loss = total_loss / self.len_epoch
        # total_train_metrics = (total_metrics /
        #  len(self.train_data_loader)).tolist()
        # add adversarial loss to custom metrics
        # metr = self.get_metrics_dic(total_train_metrics)
        self.logger.log_epoch(epoch - 1, "train",
                              total_train_loss,
                              {},
                              self.lrates)
        log = {
            "loss": total_train_loss,
            # "train_metrics": total_train_metrics
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
        losses = []
        # mean = torch.Tensor(np.array([0.485, 0.456, 0.406])[
        # :, np.newaxis, np.newaxis])
        # mean = mean.expand(3, 32, 32).to(self.device)
        # std = torch.Tensor(np.array([0.229, 0.224, 0.225])[
        #  :, np.newaxis, np.newaxis])
        # std = std.expand(3, 32, 32).to(self.device)
        # train several times on the same batch
        for _ in range(self.batch_iterations):
            pert = torch.autograd.Variable(
                self.get_perturbation()[0:data.size(0)],
                requires_grad=True).to(self.device)
            inpt = data + pert
            inpt.clamp_(0.0, 1.0)
            # inpt.sub_(mean).div_(std)
            output = self.model(inpt)
            loss = self.loss(output, target)
            losses.append(loss.item())
            loss.backward()
            self.set_perturbation(data.size(0), pert.grad)
            self.optimizer.zero_grad()
            self.optimizer.step()
        # print(losses)
        self.set_perturbation(data.size(0))
        return sum(losses)/float(len(losses))
        # if eval_metrics is True:
        #     metrics = self.eval_metrics(output, adv_output, target)
        #     return loss.item(), adv_loss.item(), metrics, \
        #         self.get_metrics_dic(metrics)
        # else:
        #     return loss.item(), adv_loss.item()

    def _validate_and_test(self, epoch, log):
        """Run validation and testing"""
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.do_testing and epoch % self.test_epochs_interval == 0:
            test_log = self._test_epoch(epoch)
            log.update(test_log)

    def get_perturbation(self):
        """Returns perturbation for each batch
        In some cases the perturbation is new for each batch,
        in others it is summed over all batches
        """
        if self.new_perturbation is True:
            self._init_perturbation()
        return self.perturbation

    def set_perturbation(self, max_size, grad=None):
        """Accumulates ascending gradients to perturbation
        :max_size: maximum length of perturbation
        :param grad: parameter gradients
        """
        if grad is not None:
            fgsm = self.eps*torch.sign(grad)
            self.perturbation[0:max_size] += fgsm.data
            self.perturbation.clamp(-self.eps, self.eps)
        else:
            self._init_perturbation()