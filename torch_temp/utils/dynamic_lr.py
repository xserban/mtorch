"""This module changes the learening rate dynamically, during training"""


class DynamicLR():
    def __init__(self, epochs, lr_values):
        """
        :param epochs: a list of epochs at which the
          learning rate will be adapted
        :param lr_values: a list of learning rates
          corresponding to the epochs
        """
        assert len(epochs) == len(lr_values)

        self.change_epochs = epochs
        self.lr_values = lr_values

        self.adapt = True
        self.current_epochs_index = 0

    def adapt_lr(self, epoch, optimizer):
        """Changes learning rate to saved value
        after a number of epochs
        :param epoch: current epoch
        :param optimizer: reference to optimizer using lr
        """
        if not self.adapt:
            return

        if epoch == self.change_epochs[self.current_epochs_index]:
            print("[INFO] \t Setting Learning Rate Value to {}".format(
                self.lr_values[self.current_epochs_index]))
            for g in optimizer.param_groups:
                g["lr"] = self.lr_values[self.current_epochs_index]

            self.increment_index()

    def increment_index(self):
        """Increments current epochs index or stops adaptation"""
        if len(self.change_epochs) > self.current_epochs_index + 1:
            self.current_epochs_index += 1
        else:
            self.adapt = False

    def still_adapting(self):
        """ Returns a true flag if still waiting to adapt a lr
        or false if all learning rates were adapted
        :returns: boolean
        """
        return self.adapt
