from abc import abstractmethod
import torch
from numpy import inf
import numpy as np

import torch_temp.train.schedulers as all_sch


class BaseTrainer:
    """
    Base class for all trainers (coaches)
    """

    def __init__(self, model, loss, metrics, optimizer, config):
        self.config = config
        self.py_logger = config.get_logger(
            "training_log", config["training"]["log_verbosity"])
        self.logger = config.logger
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
        # not improved variable
        self.not_improved = 0
        # init learning rate schedulers
        self.init_schedulers(config)

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
        self.logger.start_loops()
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log = self._log_epoch(log, result)

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            improved, best = self._save_model(log)
            if improved is False:
                break
            self._save_checkpoint(epoch, save_best=best)

        self.logger.stop_loops()
        # save best model artifacts
        if self.monitor != "off":
            self.logger.log_artifact(
                self.config.save_dir/"config.json", "config.json")
            self.logger.log_artifact(
                self.config.save_dir/"model_best.pth", "model_best.pth")

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
            if key == "train_metrics":
                log.update({mtr.get_name(): value[i]
                            for i, mtr in enumerate(self.metrics)})
            elif key == "val_metrics":
                log.update({"val_" + mtr.get_name(): value[i]
                            for i, mtr in enumerate(self.metrics)})
            elif key == "test_metrics":
                log.update({"test_" + mtr.get_name(): value[i]
                            for i, mtr in enumerate(self.metrics)})
            else:
                log[key] = value

        # print logged informations to the screen
        print("[INFO][METRICS] \t Evaluation:")
        for key, value in log.items():
            self.py_logger.info("    {:15s}: {}".format(str(key), value))
        return log

    def _save_model(self, log):
        """Decide if we save the model after evaluation of not
        :param log: dictionary with metrics
        """
        best = False
        imprvd = True
        if self.mnt_mode != "off":
            try:
                # check whether model performance
                # improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == "min" and
                            log[self.mnt_metric] <= self.mnt_best) or \
                    (self.mnt_mode ==
                     "max" and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.py_logger.warning("Warning: Metric {} is not found. "
                                       "Model performance monitoring "
                                       "is disabled.".format(self.mnt_metric))
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                print("[INFO][METRICS] \t Early Stop Metric Improved. "
                      "Setting flag to 0.")
                self.not_improved = 0
                best = True
            else:
                print("[INFO][METRICS] \t Early Stop Metric not Improved. "
                      "Incrementing flag.")
                self.not_improved += 1

            if self.not_improved > self.early_stop:
                self.py_logger.info("Validation performance "
                                    "didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.early_stop))
                imprvd = False

        return imprvd, best

    ###
    # Setup Helpers
    ###
    def _prepare_gpu_devices(self, config):
        """Configures GPU(s)"""
        print("PREPARING GPUs")
        custom_gpu = config["custom_gpu"]
        multiple_gpus = config["multiple_gpus"]

        n_gpu = torch.cuda.device_count()
        n_gpu_use, list_ids, device = 0, [], None

        if (custom_gpu["do"] is True or multiple_gpus["do"] is True) \
                and n_gpu == 0:
            self.py_logger.warning("[WARN] \t No GPU "
                                   "available on this machine, "
                                   "training will be performed on CPU.")
            n_gpu_use = 0
            device = torch.device("cpu")
        elif custom_gpu["do"] is True:
            if custom_gpu["id"] > n_gpu-1:
                self.py_logger.warning("[WARN] \t GPU id is higher"
                                       " than the max. number of GPUs."
                                       " Trying to run on the first GPU.")
                n_gpu_use = 1
                device = torch.device("cuda", 0)
            else:
                n_gpu_use = 1
                device = torch.device("cuda", custom_gpu["id"])
        elif multiple_gpus["do"] is True:
            if multiple_gpus["nr_gpus"] > n_gpu:
                self.py_logger.warning("[WARN] \t Warning: The number of "
                                       "GPU\'s configured to use is {}, "
                                       "but only {} are available "
                                       "on this machine."
                                       .format(multiple_gpus["nr_gpus"],
                                               n_gpu))
            n_gpu_use = n_gpu
            list_ids = list(range(n_gpu_use))
            device = torch.device("cuda", 0)

        return device, list_ids

    def _configure_gpu(self, model, config):
        """Configure GPUs and send the model to device
        :param model:
        :param config: config json obj.
        """
        self.device, device_ids = self._prepare_gpu_devices(
            config["host"]["gpu_settings"])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def _configure_trainer(self, config):
        """Configures custom flags for the trainer and tensorflow log
        :param config: json config obj.
        """
        # configure trainer
        self.cfg_trainer = config["training"]
        self.epochs = self.cfg_trainer["epochs"]
        self.save_period = self.cfg_trainer["save_period"]
        self.monitor = self.cfg_trainer.get("monitor", "off")
        # configure test flags
        self.test_epochs_interval = config["testing"]["test_epochs_interval"]

    def _configure_monitor(self, config):
        """Configure performance monitor and save best model
        :param config: config json obj.
        """
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    ###
    # Checkpoint helpers
    ###
    def _save_checkpoint(self, epoch, save_best=False):
        """Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the s
          aved checkpoint to 'model_best.pth'
        """
        # always save best model
        if save_best:
            state = self._get_checkpoint_data(epoch)
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.py_logger.info(
                "[INFO][TRAIN] \t Saving current best: model_best.pth ...")
        if epoch % self.save_period == 0:
            state = self._get_checkpoint_data(epoch)
            filename = str(self.checkpoint_dir /
                           "checkpoint-epoch{}.pth".format(epoch))
            torch.save(state, filename)
            self.py_logger.info(
                "[INFO][TRAIN] \t Saving checkpoint: {} ...".format(filename))

    def _get_checkpoint_data(self, epoch):
        arch = type(self.model).__name__
        return {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config.config
        }

    def _resume_checkpoint(self, resume_path):
        """Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.py_logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.py_logger.warning("[WARN] \t Warning: Architecture "
                                   "configuration given in config file "
                                   "is different from that of checkpoint."
                                   "This may yield an exception while "
                                   "state_dict is being loaded.")
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only
        # when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["opt"]["type"] != \
                self.config["optimizer"]["opt"]["type"]:
            self.py_logger.warning("[WARN] \t Warning: Optimizer type "
                                   " given in config file is different "
                                   "from that of checkpoint. "
                                   "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.py_logger.info(
            "Checkpoint loaded. Resume "
            "training from epoch {}".format(self.start_epoch))

    ###
    # Measurements Helpers
    ###
    def eval_metrics(self, output, target):
        """Evaluates all metrics"""
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] = metric.forward(output, target)

        return acc_metrics

    def get_metrics_dic(self, metrics):
        """Converts metrics to dictionary"""
        dic_metrics = {}
        for i, metric in enumerate(self.metrics):
            dic_metrics[metric.get_name()] = metrics[i]
        return dic_metrics

    ###
    # Learning rate schedler helpers
    ###
    def init_schedulers(self, config):
        self.schedulers = []
        if "lr_schedulers" in config["optimizer"] \
                and len(config["optimizer"]["lr_schedulers"]) > 0:
            self.schedulers = [self.init_scheduler(
                s) for s in config["optimizer"]["lr_schedulers"]]
            # order by priority and set the first one active
            self.schedulers = sorted(self.schedulers, key=lambda x: x.priority)
            self.schedulers[0].active = True

    def init_scheduler(self, scheduler):
        return getattr(all_sch, scheduler["type"])(
            optimizer=self.optimizer, **scheduler["args"])

    def adapt_lr(self, epoch):
        """Adapts learning rate dynamically or as scheduled
        The dynamic scheduler takes priority over the lr_scheduler
        """
        if len(self.schedulers) > 0:
            # get active scheduler
            for index, sch in enumerate(self.schedulers):
                if sch.active:
                    sch.step(epoch)
                    # activate next scheduler if
                    # after this step the scheduler is false
                    if sch.active is False:
                        if len(self.schedulers) >= index+1:
                            self.schedulers[index+1].active = True
                        del self.schedulers[index]
            self.lrates = self.get_lrates()

    def get_lrates(self):
        """Returns learning rates for all parameter groups of the optimizer"""
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group["lr"])
        return lrs
