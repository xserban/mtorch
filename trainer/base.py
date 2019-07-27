import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        # setup GPU device if available, move model into configured device
        self._configure_gpu(model, config)
        # configure trainer
        self._configure_trainer(config)
        # configure monitor
        self._configure_monitor(config)

    @abstractmethod
    def _train_epoch(self, epoch):
        """Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _test_epoch(self, epoch):
        """Run test code after training
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """Full training logic"""
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log = self._log_epoch(log, result)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            improved, best = self._save_model(log)
            if improved is False:
                break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    ###
    # Train helpers
    ###
    def _log_epoch(self, log, result):
        """Logs the information about an epoch
        :param log: dictionary with epoch
        :param result: dictionary with metrics
        :return log: dictionary with key values log metrics
        """
        for key, value in result.items():
            if key == 'metrics':
                log.update({mtr.get_name(): value[i] for i, mtr in enumerate(self.metrics)})
            elif key == 'val_metrics':
                log.update({'val_' + mtr.get_name(): value[i] for i, mtr in enumerate(self.metrics)})
            elif key == 'test_metrics':
                log.update({'test_' + mtr.get_name(): value[i] for i, mtr in enumerate(self.metrics)})
            else:
                log[key] = value

        # print logged informations to the screen
        print('[INFO] \t Evaluation:')
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
        return log

    def _save_model(self, log):
        """Decide if we save the model after evaluation of not
        :param log: dictionary with metrics

        """
        best = False
        imprvd = True
        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                    (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.logger.warning("Warning: Metric '{}' is not found. "
                                    "Model performance monitoring is disabled.".format(self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False
                not_improved_count = 0

            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                imprvd = False

        return imprvd, best

    ###
    # Setup Helpers
    ###
    def _prepare_gpu_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("[WARN] \t Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("[WARN] \t Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _configure_gpu(self, model, config):
        """Configure GPUs and send the model to device
        :param model:
        :param config: config json obj.
        """
        self.device, device_ids = self._prepare_gpu_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def _configure_trainer(self, config):
        """Configures custom flags for the trainer and tensorflow log
        :param config: json config obj.
        """
        # configure trainer
        self.cfg_trainer = config['trainer']
        self.epochs = self.cfg_trainer['epochs']
        self.save_period = self.cfg_trainer['save_period']
        self.monitor = self.cfg_trainer.get('monitor', 'off')
        # configure log flags
        self.log_index_batches = self.cfg_trainer['tensorboard_logs']['index_batches']
        self.log_params = self.cfg_trainer['tensorboard_logs']['log_params']
        self.log_train_images = self.cfg_trainer['tensorboard_logs']['log_train_images']
        self.log_test_images = self.cfg_trainer['tensorboard_logs']['log_test_images']
        # configure test flags
        self.test_epochs_interval = config['test']['test_epochs_interval']
        # instantiate Tensorboard writer
        self.writer = TensorboardWriter(config.log_dir, self.logger, self.cfg_trainer['tensorboard'])

    def _configure_monitor(self, config):
        """Configure performance monitor and save best model
        :param config: config json obj.
        """
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    ###
    # Checkpoint helpers
    ###
    def _save_checkpoint(self, epoch, save_best=False):
        """Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("[INFO] \t Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("[INFO] \t Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("[WARN] \t Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("[WARN] \t Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
