"""This module sets predefined values for the learning rate
at given steps. Instead of decaying the learning rate, this
module allows to set a value at a given epoch
"""

from .base import BaseScheduler


class DynamicLRScheduler(BaseScheduler):
    def __init__(self, optimizer, epochs, lr_values, priority=0, active=False):
        """
        :param epochs: a list of epochs at which the
          learning rate will be adapted
        :param lr_values: a list of learning rates
          corresponding to the epochs
        """
        assert len(epochs) == len(lr_values)
        super().__init__(priority, active)

        self.optimizer = optimizer
        self.change_epochs = epochs
        self.lr_values = lr_values
        self.current_epochs_index = 0

    def step(self, epoch):
        """Changes learning rate to saved value
        after a number of epochs
        :param epoch: current epoch
        """
        if self.active is False:
            return

        if epoch == self.change_epochs[self.current_epochs_index]:
            print("[INFO][OPTIMIZER] \t Setting Learning Rate Value to {}".format(
                self.lr_values[self.current_epochs_index]))
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr_values[self.current_epochs_index]

            self.increment_index()

    def increment_index(self):
        """Increments current epochs index or stops adaptation"""
        if len(self.change_epochs) > self.current_epochs_index + 1:
            self.current_epochs_index += 1
        else:
            self.active = False
