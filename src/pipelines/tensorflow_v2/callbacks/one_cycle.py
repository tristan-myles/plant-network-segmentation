# ================================== Credit: ==================================
# https://www.kaggle.com/robotdreams/one-cycle-policy-with-keras
# =============================================================================

from typing import Tuple

import matplotlib.pyplot as plt
import tensorflow as tf


class OneCycleLR(tf.keras.callbacks.Callback):

    def __init__(self, init_lr: float, max_lr: float, final_tail_lr: float,
                 iterations: int, tail_length: float, init_momentum: float =
                 None, min_momentum: float = None, cyclic_momentum: bool =
                 False):
        """
        Instantiator for the Leslie Smith's 1cylce Policy

        :param init_lr: initial learning rate
        :param max_lr: maximum learning rate, i.e. the cycle peak
        :param final_tail_lr: minimum learning rate, to be used in the tail
        phase once the cycle is complete
        :param iterations: total number of iterations to run the cycle,
        i.e. number of batches * epochs
        :param tail_length: the tail length expressed as a percentage of the
         step size, i.e larger value => a larger tail length and a smaller
         step size
        :param init_momentum: initial momentum
        :param min_momentum: minimum momentum
        :param cyclic_momentum: whether to use cyclic momentum

        .. Note:: Only certain optimisers require momentum, for example SGD
        """
        super().__init__()
        self.lr_list = []
        self.steps_list = []

        self.init_lr = init_lr
        self.max_lr = max_lr

        self.step_size = round(iterations / (2 + tail_length))
        self.full_cycle = self.step_size * 2
        self.tail_length = iterations - self.full_cycle

        # Counter to know when step_size is done:
        self.iteration_counter = 0

        # Need to exclude first case where lr is at baseline
        self.increment = (max_lr - self.init_lr) / self.step_size

        # Note: negative
        self.tail_decrement = (final_tail_lr - init_lr) / self.tail_length

        self.cyclic_momentum = cyclic_momentum

        # Momentum
        if cyclic_momentum:
            self.momentum_list = []

            if init_momentum and min_momentum:
                self.init_momentum = init_momentum
                self.min_momentum = min_momentum

                self.momentum_decrement = ((min_momentum - self.init_momentum)
                                           / self.step_size)
            else:
                raise ValueError(
                    "Please provide both an initial and a final momentum value"
                    " when setting the cyclic momentum flag to true")

    def get_lr(self, iteration: int) -> float:
        """
        Gets the updated learning rate, where both the size and direction of
        change depends on where the current iterations falls in the LR cycle

        :param iteration: current iteration number
        :return: updated learning rate
        """
        cycle_perc = iteration / self.full_cycle

        # Normal cycle
        # Increase
        if cycle_perc <= 0.5:
            # starts @ 0 for baseline LR
            lr = self.init_lr + iteration * self.increment

        # Decrease
        elif 0.5 < cycle_perc <= 1:
            lr = self.init_lr + (self.full_cycle - iteration) * self.increment

        # Tail cycle | cycle_perc > 1
        else:
            lr = self.init_lr + (
                    iteration - self.full_cycle) * self.tail_decrement

        return lr

    def get_momentum(self, iteration: int) -> float:
        """
        Gets the updated momentum, where both the size and direction of
        change depends on where the current iterations falls in the LR cycle

        :param iteration: current iteration number
        :return: updated momentum
        """
        cycle_perc = iteration / self.full_cycle

        # Normal cycle
        # Increase
        if cycle_perc <= 0.5:
            # starts @ 0 for baseline momentum
            momentum = self.init_momentum + iteration * self.momentum_decrement

        # Decrease
        elif 0.5 < cycle_perc <= 1:
            momentum = self.init_momentum + (
                    self.full_cycle - iteration) * self.momentum_decrement

        # Tail cycle | cycle_perc > 1
        else:
            momentum = self.init_momentum

        return momentum

    def on_train_begin(self, batch: int, logs={}) -> None:
        """
        Overwrite the optimiser LR and momentum (if applicable) to what's
        specified in this callback

        :param batch: batch number
        :param logs: TF logs
        """
        print(f"Updating LR to callback init LR: {self.init_lr}")
        self.model.optimizer.lr.assign(self.init_lr)

        if self.cyclic_momentum:
            print(f"Updating momentum to callback init momentum:"
                  f" {self.init_momentum}")
            self.model.optimizer.momentum.assign(self.init_momentum)

    def on_train_batch_end(self, batch: int, logs={}) -> None:
        """
         Updates LR and momentum (if applicable)

        :param batch: batch number
        :param logs: TF logs
        """
        self.steps_list.append(self.iteration_counter)
        self.lr_list.append(self.model.optimizer.lr.read_value().numpy())

        # setting iteration count for the next batch... => batch + 1
        self.iteration_counter += 1

        self.model.optimizer.lr.assign(self.get_lr(self.iteration_counter))

        # update momentum if flag
        if self.cyclic_momentum:
            self.momentum_list.append(
                self.model.optimizer.momentum.read_value().numpy())
            self.model.optimizer.momentum.assign(
                self.get_momentum(self.iteration_counter))

    @staticmethod
    def get_iterations(train_length: int, batch_size: int, epochs: int) -> int:
        """
         Gets the number of iterations required to achieve the maximum
        learning rate, provided both the initial learning rate
        and the increment size (which is on a log base 10 scale)

        :param train_length: sample size of training input (n)
        :param batch_size: batch size
        :param epochs: number of epochs
        :return: the total number of iterations the optimiser will run for

        .. Note:: Integer division is used
        """
        return train_length // batch_size * epochs

    def plot_ocp_lr(self, figsize: Tuple[int, int] = (5, 5),
                    ax: plt.axes = None) -> plt.axes:
        """
        Plots the instance lr, plot should show the expected 1cycle
        lr shape

        :param figsize:  matplotlib.pyplot figure size
        :param ax:  matplotlib.pyplot axis
        :return: matplotlib.pyplot axis containing the plot described above
        """

        if not ax:
            fig, ax = plt.subplots(1, figsize=figsize)
            fig.suptitle("One Cycle LR with Tail")

        ax.plot(self.steps_list, self.lr_list, label="1cycle LR")

        ax.set_xlabel("Steps")
        ax.set_ylabel("Learning Rate")

        return ax

    def plot_ocp_momentum(self, figsize: Tuple[int, int] = (5, 5),
                          ax: plt.axes = None):
        """
        Plots the instance momentum, plot should show the expected 1cycle
        momentum shape

        :param figsize:  matplotlib.pyplot figure size
        :param ax:  matplotlib.pyplot axis
        :return: matplotlib.pyplot axis containing the plot described above
        """

        if not ax:
            fig, ax = plt.subplots(1, figsize=figsize)
            fig.suptitle("One Cycle Momentum with Tail")

        ax.plot(self.steps_list, self.momentum_list, label="1cycle Momentum")

        ax.set_xlabel("Steps")
        ax.set_ylabel("Momentum")

        return ax

    def plot_ocp_lr_momentum(self, plot_opt: int,
                             figsize: Tuple[int, int] = (10, 5)) -> None:
        """
        Wrapper that plots both the 1cycle momentum and lr together; the
        plots can either be shown side-by-side or overlaid

        :param plot_opt: plot option, 1 indicates that plots should be
        overlaid while two indicates that plots should be shown side-by-side
        :param figsize:  matplotlib.pyplot figure size
        """
        if plot_opt == 1:
            fig, ax = plt.subplots(ncols=1, figsize=figsize)

            ax = self.plot_ocp_momentum(ax=ax)

            ax = self.plot_ocp_lr(ax=ax)
            ax.set_title("One Cycle LR & Momentum with Tail")

            ax.legend()

        elif plot_opt == 2:
            fig, ax = plt.subplots(ncols=2, figsize=figsize)

            ax[0] = self.plot_ocp_momentum(ax=ax[0])
            ax[0].set_title("One Cycle Momentum with Tail")

            ax[1] = self.plot_ocp_lr(ax=ax[1])
            ax[1].set_title("One Cycle LR with Tail")

            ax[1].legend()
