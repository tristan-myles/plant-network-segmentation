# ================================== Credit: ==================================
# https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py
# https://www.kaggle.com/robotdreams/one-cycle-policy-with-keras
# https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
# =============================================================================
import math

import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple


class LRRangeTest(tf.keras.callbacks.Callback):

    def __init__(self, init_lr: float = 1e-8, max_lr: float = 10.0,
                 beta: float = 0.98, increment: float = None,
                 iterations: int = None, verbose: int = None):
        """
        Instantiator for the Leslie Smith's LRRangeTest

        :param init_lr: initial learning rate
        :param max_lr: the max learning rate to increase to
        :param beta: the parameter to be used for smoothed loss,
        larger beta => more smoothing
        :param increment: the increment to be used; the increment is on a
        log10 scale
        :param iterations: the number of iterations to run the test for
        :param verbose: number of batches after which to print the current
        lr, the default is for no output to be printed

        .. Note:: Only one of increment or iterations needs to be provided,
        if both are provided increment will be used
        """
        # use the term iterations to be consistent with LS paper
        super().__init__()
        self.batch_log_lr = []
        self.batch_nums = []
        self.smoothed_losses = []

        self.best_loss = 0
        self.avg_loss = 0

        # user defined
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.beta = beta

        self.verbose = verbose

        if increment:
            print("Note: Increment must be on the log scale")
            self.increment = increment
            self.iterations = self.get_iterations(init_lr, max_lr, increment)
        elif iterations:
            self.iterations = iterations
            self.increment = self.get_increment(init_lr, max_lr, iterations)
        else:
            raise ValueError(
                "Please provide a value for either increment or iteration")

    @staticmethod
    def get_iterations(init_lr: float, max_lr: float,
                       increment: float) -> float:
        """
        Gets the number of iterations required to achieve the maximum
        learning rate, provided both the initial learning rate
        and the increment size (which is on a log base 10 scale)

        :param init_lr: initial learning rate
        :param max_lr: maximum learning rate
        :param increment: multiplicative increment size on a log base 10 scale
        :return: number of iterations

        .. Note:: Integer division is used
        """
        return ((math.log(max_lr, 10) - math.log(init_lr, 10)) //
                math.log(increment, 10))

    @staticmethod
    def get_increment(init_lr: float, max_lr: float, iterations: int) -> float:
        """
        Gets the increment, using a log base 10 scale, required to achieve
        the maximum learning rate, provided both the initial learning rate
        and the number of iterations

        :param init_lr: initial learning rate
        :param max_lr: maximum learning rate
        :param iterations: number of iterations
        :return: increment on log base 10 scale
        """
        # steps includes step 0 at which the full rate is used => (steps - 1)
        return (max_lr / init_lr) ** (1 / (iterations - 1))

    def on_train_batch_end(self, batch, logs={}) -> None:
        """
        Updates LR and momentum (if applicable), as well as calculates
        smoothed loss at the end of each training batch

        :param batch: batch number
        :param logs: TF logs
        :return: None

        .. Note:: Batch starts at 0 and using training loss
        """
        current_loss = logs['loss']

        # if the loss = nan, stop training
        if math.isnan(current_loss):
            # self.model.stop_training not working =>
            # see: https://github.com/tensorflow/tensorflow/issues/41174
            print("\nTraining stopped, current loss = NaN")
            self.model.stop_training = True

        self.avg_loss = self.beta * self.avg_loss + (
                1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** (batch + 1))

        if smoothed_loss < self.best_loss or batch == 0:
            self.best_loss = smoothed_loss

        # stopping condition
        if batch > 0 and smoothed_loss > (self.best_loss * 4):
            print("\nTraining stopped, smoothed loss > (best loss * 4)")
            self.model.stop_training = True

        # current LR
        lr = self.model.optimizer.lr.read_value()
        self.batch_log_lr.append(math.log(lr, 10))

        # getting new LR and assigning
        lr *= self.increment
        self.model.optimizer.lr.assign(lr)

        # assign to lists for reporting purposes
        self.batch_nums.append(batch)
        self.smoothed_losses.append(smoothed_loss)

        if self.verbose:
            if batch % self.verbose == 0:
                print("\nEpoch: {} \t Learning Rate: {}".format(
                    batch, self.model.optimizer.lr.read_value()))

    def plot_LRTest(self, num_points_excl: int,
                    figsize: Tuple[int, int] = (5, 5)) -> None:
        """
        Plot the results of the LR Range test

        :param figsize: matplotlib.pyplot figure size
        :param num_points_excl: number of end points to exclude, required
         due to error with self.model.stop_training = True, present in TF 2.3.0
        :return: None

        .. Note:: Requires the instance batch_log_lr and smoothed_losses
         variables to not be empty.
        """

        fig, ax = plt.subplots(1, figsize=figsize)
        fig.suptitle("Learning Rate Range Test")

        ax.plot(self.batch_log_lr[0:-num_points_excl],
                self.smoothed_losses[0:-num_points_excl],
                label="1cycle LR")

        ax.set_xlabel("log$_{10}$ LR")
        ax.set_ylabel("Smoothed loss")
