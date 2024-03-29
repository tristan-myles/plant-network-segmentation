from typing import Tuple

import kerastuner
from kerastuner import HyperModel
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D,
                                     Conv2DTranspose, concatenate)

from src.pipelines.tensorflow_v2.losses.custom_losses import *


# *========================== U-Net Building Block ===========================*
class HyperUnet(HyperModel):
    """
    A Hyper U-Net model which can be tuned by kerastuner
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 output_channels: int):
        """
        Instantiates a HyperUnet object.

        :param input_shape: the shape of the input
        :param output_channels: the number of output channels to use in the
         final layer; this is related to the number of classes in the final
         prediction
         """
        self.input_shape = input_shape
        self.output_channels = output_channels

    def build(self, hp: kerastuner.HyperParameters) -> kerastuner.HyperModel:
        """
        A function which creates a kerastuner hyper model with a defined
        search space. This function signature matches the requirements of
        the hyperparameter tuning algorithms in kerastuner.

        :param hp: a HyperParameters instance
        :return: a HyperUnet instance
        """

        # create the search space
        initializer = hp.Choice("initializer", ["he_normal", "glorot_uniform"])
        activation = hp.Choice("activation", ["relu", "selu"])
        filter = hp.Choice("filters", [0, 1, 2, 3])
        kernel_size = hp.Choice("kernel_size", [3])
        optimizer = hp.Choice("optimizer", ["adam", "sgd"])
        padding = "same"
        loss_choice = hp.Choice("loss", ["focal", "wce", "bce"])

        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'),
                momentum=hp.Float('momentum', 0.5, 0.9))

        if initializer == "he_normal":
            initializer = tf.keras.initializers.he_normal(seed=3141)
        elif initializer == "glorot_uniform":
            initializer = tf.keras.initializers.glorot_uniform(seed=3141)

        if activation == "relu":
            activation = tf.nn.relu
        elif activation == "selu":
            activation = tf.nn.selu

        if loss_choice == "bce":
            loss = tf.keras.losses.binary_crossentropy
        elif loss_choice == "wce":
            loss = WeightedCE(hp.Float("loss_weight", 0.5, 0.99))
        elif loss_choice == "focal":
            loss = FocalLossV2(hp.Float("loss_weight", 0.5, 0.99))

        input = Input(shape=self.input_shape)

        # Down 1
        conv1 = Conv2D(filters=8 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(input)
        conv1 = Conv2D(filters=8 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(conv1)
        # Half the x,y dimensions
        pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv1)
        conv2 = Conv2D(filters=16 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(pool1)
        conv2 = Conv2D(filters=16 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(conv2)

        pool2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2)
        conv3 = Conv2D(filters=32 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(pool2)
        conv3 = Conv2D(filters=32 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(conv3)

        pool3 = MaxPool2D(pool_size=(2, 2), strides=2)(conv3)
        conv4 = Conv2D(filters=64 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(pool3)
        conv4 = Conv2D(filters=64 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(conv4)

        # bottleneck
        pool4 = MaxPool2D(pool_size=(2, 2), strides=2)(conv4)
        conv5 = Conv2D(filters=128 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(pool4)
        conv5 = Conv2D(filters=128 * 2 ** filter, kernel_size=kernel_size,
                       padding=padding, activation=activation,
                       kernel_initializer=initializer)(conv5)

        # Upblock
        uptrans1 = Conv2DTranspose(filters=64 * 2 ** filter,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=initializer)(conv5)
        skip1 = concatenate([conv4, uptrans1])
        up1 = Conv2D(filters=64 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(skip1)
        up1 = Conv2D(filters=64 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(up1)

        uptrans2 = Conv2DTranspose(filters=32 * 2 ** filter,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=initializer)(up1)
        skip2 = concatenate([conv3, uptrans2])
        up2 = Conv2D(filters=32 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(skip2)
        up2 = Conv2D(filters=32 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(up2)

        uptrans3 = Conv2DTranspose(filters=16 * 2 ** filter,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=initializer)(up2)
        skip3 = concatenate([conv2, uptrans3])
        up3 = Conv2D(filters=16 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(skip3)
        up3 = Conv2D(filters=16 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(up3)

        uptrans4 = Conv2DTranspose(filters=8 * 2 ** filter,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=initializer)(up3)
        skip4 = concatenate([conv1, uptrans4])
        up4 = Conv2D(filters=8 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(skip4)
        up4 = Conv2D(filters=8 * 2 ** filter, kernel_size=kernel_size,
                     padding=padding, activation=activation,
                     kernel_initializer=initializer)(up4)

        output_layer = Conv2D(self.output_channels, 1, strides=1,
                              padding=padding,
                              activation="sigmoid",
                              name="classification_layer")(up4)

        model = tf.keras.Model(inputs=input, outputs=output_layer,
                               name="Hyper-UNet")

        model.compile(
            optimizer=opt, loss=loss,
            metrics=['accuracy', tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall"),
                     tf.keras.metrics.AUC(name="auc", curve="PR")]
        )

        return model
