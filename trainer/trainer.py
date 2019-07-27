import numpy as np
import torch
from torchvision.utils import make_grid
from trainer.base import BaseTrainer
from tqdm import tqdm
from utils import inf_loop


class Trainer(BaseTrainer):
    """Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, train_data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None):
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
        self.lr_scheduler = lr_scheduler

    def _eval_metrics(self, output, target):
        """Evaluates all metrics and adds them to tensorboard"""
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric.forward(output, target)

        return acc_metrics

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

            The metrics in log must have the key 'metrics'.
        """
        print('[INFO] \t Starting Training Epoch {}:'.format(epoch))
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(tqdm(self.train_data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            # run batch and get loss
            loss, metrics = self._run_batch(data, target)
            total_loss += loss
            total_metrics += metrics
            # log info
            if self.log_index_batches:
                self._log_batch((epoch - 1) * self.len_epoch + batch_idx, 'train', loss, metrics)
                if self.log_train_images:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        if not self.log_index_batches:
            self._log_batch(epoch - 1, 'train',
                            total_loss / self.len_epoch,
                            (total_metrics / self.len_epoch).tolist())

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        # run validation and testing
        self._validate(epoch, log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    ###
    # Epoch helpers
    ###
    def _run_batch(self, data, target, train=True):
        """Runs batch optimization and returns loss
        :param data: input batch
        :param target: labels batch
        :return: loss value
        """
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss.forward(output, target)
        if train is True:
            loss.backward()
            self.optimizer.step()
        metrics = self._eval_metrics(output, target)
        return loss.item(), metrics

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
            The validation metrics in log must have the key 'val_metrics'.
        """
        print('[INFO] \t Starting Validation Epoch {}:'.format(epoch))
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                # get loss and run metrics
                loss, metrics = self._run_batch(data, target, train=False)
                total_val_loss += loss
                total_val_metrics += metrics
                # log results
                if self.log_index_batches:
                    self._log_batch((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid', loss, metrics)
                    if self.log_train_images:
                        self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if not self.log_index_batches:
            self._log_batch(epoch - 1, 'valid',
                            total_val_loss / len(self.valid_data_loader),
                            (total_val_metrics / len(self.valid_data_loader)).tolist())

        # add histogram of model parameters to the tensorboard
        if self.log_params:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _test_epoch(self, epoch):
        print('[INFO] \t Starting Test Epoch {}:'.format(epoch))
        self.model.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.test_data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                # get loss and and run metrics
                loss, metrics = self._run_batch(data, target, train=False)
                total_test_loss += loss
                total_test_metrics += metrics
                # log results
                if self.log_index_batches:
                    self._log_batch((epoch - 1) * len(self.test_data_loader) + i, 'test', loss, metrics)
                    if self.log_test_images:
                        self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if not self.log_index_batches:
            self._log_batch(epoch - 1, 'test',
                            total_test_loss / len(self.test_data_loader),
                            (total_test_metrics / len(self.test_data_loader)).tolist())

        return {
            'test_loss': total_test_loss / len(self.test_data_loader),
            'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
        }

    ###
    # Log helpers
    ###
    def _log_batch(self, step, env, loss, metrics):
        self.writer.set_step(step, env)
        self.writer.add_scalar('loss', loss)
        self._log_metrics(metrics)

    def _log_metrics(self, metrics):
        """Adds all metric values to tensorboard"""
        for i, metric in enumerate(self.metrics):
            self.writer.add_scalar('{}'.format(metric.get_name()), metrics[i])

    def _log_progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
