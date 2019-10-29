import numpy as np
import torch
import torch.nn.functional as F

from gensim.models import KeyedVectors

from tqdm import tqdm
from core.train.coach.base import BaseTrainer
from core.utils import inf_loop


class PrototypicalTrainer(BaseTrainer):
    """Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics,
                 optimizer, config, train_data_loader,
                 valid_data_loader=None,
                 test_data_loader=None,
                 len_epoch=None,
                 word2vec_path=""):
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

        self.text_classes = train_data_loader.get_class_names()

        # init word2vec model
        self.word2vec_model = KeyedVectors.load_word2vec_format(
            word2vec_path, binary=True)

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

        for batch_idx, (data, target) in \
                enumerate(tqdm(self.train_data_loader)):

            data, target = data.to(self.device), target.to(self.device)
            # run batch and get loss
            loss = self._run_batch(data, target)
            total_loss += loss
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
        self.logger.log_epoch(epoch - 1, "train",
                              total_train_loss,
                              {},
                              self.lrates)
        log = {
            "loss": total_train_loss,
        }
        # run validation and testing
        self._validate(epoch, log)
        self.adapt_lr(epoch)

        return log

    ###
    # Epoch helpers
    ###
    def _run_batch(self, data, target, eval_metrics=False, train=True):
        """Runs batch optimization and returns loss
        :param data: input batch
        :param target: labels batch
        :return: loss value
        """
        vec_target = torch.tensor(self._get_proto_targets(
            target)).to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss(output, vec_target)
        if train is True:
            loss.backward()
            self.optimizer.step()
        if eval_metrics is True:
            classes = self._get_normal_targets(output)
            metrics = self.eval_metrics(classes, target)
            return loss.item(), metrics, self.get_metrics_dic(metrics)
        else:
            return loss.item()

    def _validate(self, epoch, log):
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
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                # get loss and run metrics
                loss, metrics, dic_metrics = self._run_batch(
                    data, target, eval_metrics=True, train=False)
                total_val_loss += loss
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
        total_loss = total_val_loss / len(self.valid_data_loader)
        total_metrics = (total_val_metrics /
                         len(self.valid_data_loader)).tolist()
        self.logger.log_epoch(epoch - 1, "valid",
                              total_loss,
                              self.get_metrics_dic(total_metrics),
                              None)
        # add histogram of model parameters to the tensorboard
        self.logger.log_validation_params(
            epoch-1, "valid", self.model.named_parameters())
        # return final log metrics
        return {
            "val_loss": total_loss,
            "val_metrics": total_metrics
        }

    def _test_epoch(self, epoch):
        print("[INFO][TEST] \t Starting Test Epoch {}:".format(epoch))
        self.model.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.test_data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                # get loss and and run metrics
                loss, metrics, dic_metrics = self._run_batch(
                    data, target, eval_metrics=True, train=False)
                total_test_loss += loss
                total_test_metrics += metrics
                # log results specific to batch
                self.logger.log_batch((epoch - 1) *
                                      len(self.test_data_loader) + i,
                                      "test",
                                      loss,
                                      dic_metrics,
                                      data)
        # log results specific to epoch
        total_loss = total_test_loss / len(self.test_data_loader)
        total_metrics = (total_test_metrics /
                         len(self.test_data_loader)).tolist()
        self.logger.log_epoch(epoch - 1, "test",
                              total_loss,
                              self.get_metrics_dic(total_metrics),
                              None)
        # return final log metrics
        return {
            "test_loss": total_loss,
            "test_metrics": total_metrics
        }

    def _get_proto_targets(self, target):
        """Returns word2vec representation for given targets
        :param target:
        """
        text_target = [self.text_classes[i] for i in target]
        vec_target = [self.word2vec_model.wv[t] for t in text_target]
        return vec_target

    def _get_normal_targets(self, batch):
        """Converts nn output to class
        :param batch: batch of output vectors
        """
        classes = []
        for _, output_vector in enumerate(batch):
            class_vec = torch.tensor([self.word2vec_model.wv[t]
                                      for t in self.text_classes]).to(self.device)
            best = 10
            index = 0
            for j, vec in enumerate(class_vec):
                sim = 1 - \
                    F.cosine_similarity(output_vector, vec, 0)
                if sim < best:
                    best = sim
                    index = j
            classes.append(index)
        return torch.tensor(classes).to(self.device)

    def eval_metrics(self, output, target):
        """Evaluates all metrics"""
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] = metric.forward(output, target)

        return acc_metrics
