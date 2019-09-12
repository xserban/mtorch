import torch

from art.classifiers import PyTorchClassifier
from art.attacks import FastGradientMethod
from art.data_generators import PyTorchDataGenerator

from tqdm import tqdm
from torch_temp.trainer.base import BaseTrainer


class AdversarialTrainer(BaseTrainer):
    """Performs adversarial training
       using a given attack
    """

    def __init__(self, model, loss, metrics, optimizer, config,
                 train_data_loader,
                 foolbox_model_params={},
                 attack_params={},
                 valid_data_loader=None,
                 test_data_loader=None,
                 lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config)

        self.config = config
        self.train_data_loader = train_data_loader

        # epoch-based training
        self.len_epoch = len(self.train_data_loader)

        # initialize adversarial objects
        self._init_adversarial_loaders(train_data_loader,
                                       valid_data_loader,
                                       test_data_loader)
        self._init_adversarial_classifier()
        self._init_adversarial_attack()

        self.do_validation = self.valid_data_loader is not None
        self.do_testing = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.foolbox_model_params = foolbox_model_params
        self.attack_params = attack_params

    def _init_adversarial_classifier(self, *args, **kwargs):
        """Initializes adversarial toolbox classifier"""
        # TODO: implement hyperparams parsing
        self.adversarial_model = PyTorchClassifier(model=self.model,
                                                   clip_values=(
                                                       0, 1),
                                                   loss=self.loss,
                                                   optimizer=self.optimizer,
                                                   input_shape=(1, 28, 28),
                                                   nb_classes=10)

    def _init_adversarial_attack(self):
        """Initializes adversarial attack"""
        # TODO: Implement param parsing
        self.attack = FastGradientMethod(classifier=self.adversarial_model,
                                         eps=0.2)

    def _init_adversarial_loaders(self, train_loader,
                                  valid_loader=None,
                                  test_loader=None):
        """Initializes adversarial data loaders for training, validation
           and testing
        :param train_loader: pytorch data loader for training data
        :param valid_loader: pytorch data loader for validation data
        :param test_loader: pytorch data loader for test data
        """
        self.adv_train_loader = PyTorchDataGenerator(train_loader,
                                                     len(train_loader),
                                                     self.config.train_data_loader.args.batch_size)
        if valid_loader is not None:
            self.adv_valid_loader = PyTorchDataGenerator(valid_loader,
                                                         len(valid_loader),
                                                         self.config.train_data_loader.args.batch_size)
        if test_loader is not None:
            self.adv_test_loader = PyTorchDataGenerator(valid_loader,
                                                        len(valid_loader),
                                                        self.config.test.test_batch_size)

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
        print('[INFO] \t Starting Adversarial Training Epoch {}:'.format(epoch))

        # self.model.train()
        # total_loss = 0

        # for batch_idx, (data, target) in \
        #         enumerate(tqdm(self.train_data_loader)):

        #     data, target = data.to(self.device), target.to(self.device)
        #     # run batch and get loss
        #     loss = self._run_batch(data, target)
        #     total_loss += loss
        #     # log info specific to this batch
        #     self.logger.log_batch((epoch - 1) * self.len_epoch + batch_idx,
        #                           'train',
        #                           loss,
        #                           {},
        #                           data)

        #     if batch_idx == self.len_epoch:
        #         break
        # # log info specific to the whole epoch
        # total_train_loss = total_loss / self.len_epoch
        # self.logger.log_epoch(epoch - 1, 'train',
        #                       total_train_loss,
        #                       {})

        # log = {
        #     'loss': total_train_loss,
        # }
        # # run validation and testing
        # self._validate(epoch, log)
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()

        # return log

    ###
    # Epoch helpers
    ###

    def _run_batch(self, data, target, eval_metrics=False, train=True):
        """Runs batch optimization and returns loss
        :param data: input batch
        :param target: labels batch
        :return: loss value
        """
        self.optimizer.zero_grad()
        # Generate adversarial data
        print('aaa', data[0].shape)
        classifier = PyTorchClassifier(model=self.model,
                                       clip_values=(
                                           0, 1),
                                       loss=self.loss,
                                       optimizer=self.optimizer,
                                       input_shape=data[0].shape,
                                       nb_classes=10)
        attack = FastGradientMethod(classifier=classifier, eps=0.2)
        adversarial_data = attack.generate(x=data)
        # TODO: decide if we should only train on adversarial data
        # or on both normal and adversarial data

        # Train on adversarial data
        output = self.model(adversarial_data)
        loss = self.loss.forward(output, target)
        if train is True:
            loss.backward()
            self.optimizer.step()
        if eval_metrics is True:
            metrics = self._eval_metrics(output, target)
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
                loss, metrics, dic_metrics = self._run_batch(
                    data, target, eval_metrics=True, train=False)
                total_val_loss += loss
                total_val_metrics += metrics
                # log results specific to batch
                self.logger.log_batch((epoch - 1) *
                                      len(self.valid_data_loader) +
                                      batch_idx,
                                      'valid',
                                      loss,
                                      dic_metrics,
                                      data)
        # log info specific to the whole validation epoch
        total_loss = total_val_loss / len(self.valid_data_loader)
        total_metrics = (total_val_metrics /
                         len(self.valid_data_loader)).tolist()
        self.logger.log_epoch(epoch - 1, 'valid',
                              total_loss,
                              self.get_metrics_dic(total_metrics))
        # add histogram of model parameters to the tensorboard
        self.logger.log_validation_params(
            epoch-1, 'valid', self.model.named_parameters())
        # return final log metrics
        return {
            'val_loss': total_loss,
            'val_metrics': total_metrics
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
                loss, metrics, dic_metrics = self._run_batch(
                    data, target, eval_metrics=True, train=False)
                total_test_loss += loss
                total_test_metrics += metrics
                # log results specific to batch
                self.logger.log_batch((epoch - 1) *
                                      len(self.test_data_loader) + i,
                                      'test',
                                      loss,
                                      dic_metrics,
                                      data)
        # log results specific to epoch
        total_loss = total_test_loss / len(self.test_data_loader)
        total_metrics = (total_test_metrics /
                         len(self.test_data_loader)).tolist()
        self.logger.log_epoch(epoch - 1, 'test',
                              total_loss,
                              self.get_metrics_dic(total_metrics))
        # return final log metrics
        return {
            'test_loss': total_loss,
            'test_metrics': total_metrics
        }
