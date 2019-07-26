import numpy as np
import torch
import torch.nn.functional as F

from base import BaseTrainer
from utils import inf_loop

from tqdm import tqdm


class MetaTrainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_valdation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _train_epoch(self, epoch):
        pass

    def _meta_epoch(self, x_spt, y_spt, x_qry, y_qry):
        task_num, set_size, c_, h, w = x_spt.size()
        query_size = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # run the task and compute loss for k=0
            output = self.model(x_spt[i], vars=None, bn_training=True)
            loss = self.loss(output, y_spt[i])
            # get gradients and perform update
            gradients = torch.autograd.grad(loss, self.model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(gradients, self.model.parameters())))

            # compute loss and accuracy before updating the parameters for this task
            loss_q, correct_q = self._get_metrics(x_qry[i], y_qry[i], self.model.parameters())
            losses_q[0] += loss_q
            corrects[0] += correct_q

            # compute loss and accuracy after updating the parameters
            loss_q, correct_q = self._get_metrics(x_qry[i], y_qry[i], fast_weights)
            losses_q[1] += loss
            corrects[1] += correct_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~k-1
                output = self.model(x_spt[i], fast_weights, bn_training=True)
                loss = self.loss(output, y_spt[i])
                # compute grad on theta_pi
                gradients = torch.autograd.grad(loss, fast_weights)
                # update
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(gradients, fast_weights)))

                # get loss and save it
                output_q = self.model(x_qry[i], fast_weights, bn_training=True)
                loss_q = self.loss(output_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(output_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks - optimize
        loss_q = losses_q[-1] / task_num
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

    def _get_metrics(self, x, y, params):
        with torch.no_grad():
            output_q = self.net(x, params, bn_training=True)
            loss_q = self.loss(output_q, y)

            pred_q = F.softmax(output_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y).sum().item()

            return loss_q, correct
