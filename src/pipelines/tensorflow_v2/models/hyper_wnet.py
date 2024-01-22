# W-Net structure copied from the paper:
# The Little W-Net That Could: State-of-the-Art Retinal Vessel Segmentation
# with Minimalistic Models by Galdran et al.
# Link: https://arxiv.org/abs/2009.01907
from typing import Tuple

import kerastuner
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Conv2DTranspose,
                                     Input, BatchNormalization, concatenate)

from src.pipelines.tensorflow_v2.losses.custom_losses import *
from src.pipelines.tensorflow_v2.models.hyper_unet_resnet import ResBlock


# *============================ Conv Bridge Block ============================*
class ConvBridgeBlock(tf.keras.layers.Layer):
    """
    A Convolutional Bridge Block to be used in a W-Net. This is similar to the
    class defined in the wnet.py script, but has the addition of the
    kernel_size in the instantiator to allow the option to include this
    parameter in the search
    """

    def __init__(self,
                 channels,
                 kernel_size,
                 initializer,
                 activation):
        """
        Instantiates a ConvBridgeBlock object

        :param channels: number of filters required for the conv layers
        :param kernel_size: the side length of a kernel
        :param activation: the activation function to use
        :param initializer: the weight initializer to use
        """

        super().__init__()

        self.conv = Conv2D(filters=channels, kernel_size=kernel_size,
                           strides=1,
                           padding='same',
                           kernel_initializer=initializer)
        self.bn = BatchNormalization()
        self.activation = activation

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies a ConvBridgeBlock to the an input

        :param x: the input to apply the ConvBridgeBlock to
        :return: the output of the ConvBridgeBlock
        """
        x1 = self.conv(x)
        x1 = self.activation(x1)
        x1 = self.bn(x1)

        return x1


# *================================== W-Net ==================================*
class HyperWnet(kerastuner.HyperModel):
    """
    A Hyper W-Net model which can be tuned by kerastuner
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 output_channels: int):
        """
        Instantiates a HyperWnet object.

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
        :return: a HyperWnet instance
        """
        # Search space
        # create the search space
        initializer = hp.Choice("initializer", ["he_normal", "glorot_uniform"])
        filter = hp.Choice("filters", [0, 1, 2])
        kernel_size = hp.Choice("kernel_size", [3])
        optimizer = hp.Choice("optimizer", ["adam", "sgd"])
        wnets = hp.Choice("num_wnets", [2])
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

        if loss_choice == "bce":
            loss = tf.keras.losses.binary_crossentropy
        elif loss_choice == "wce":
            loss = WeightedCE(hp.Float("loss_weight", 0.5, 0.99))
        elif loss_choice == "focal":
            loss = FocalLossV2(hp.Float("loss_weight", 0.5, 0.99))

        activation = tf.nn.relu

        input = []
        res_down1 = []
        pool1 = []
        res_down2 = []
        pool2 = []
        res_bottle = []
        trans_conv1 = []
        res_up1 = []
        bridge1 = []
        concat1 = []
        trans_conv2 = []
        res_up2 = []
        bridge2 = []
        concat2 = []
        output_layer = []

        input.append(Input(shape=self.input_shape))

        for i in range(wnets):
            # Mini-Unet
            res_down1.append(ResBlock(8 * 2 ** filter,
                                      kernel_size=kernel_size,
                                      activation=activation,
                                      initializer=initializer,
                                      decode=True)(input[i]))
            pool1.append(MaxPool2D(pool_size=2, strides=2)(res_down1[i]))

            res_down2.append(ResBlock(16 * 2 ** filter,
                                      kernel_size=kernel_size,
                                      activation=activation,
                                      initializer=initializer,
                                      decode=True)(pool1[i]))
            pool2.append(MaxPool2D(pool_size=2, strides=2)(res_down2[i]))

            res_bottle.append(ResBlock(32 * 2 ** filter,
                                       kernel_size=kernel_size,
                                       activation=activation,
                                       initializer=initializer,
                                       decode=True)(pool2[i]))

            # Expanding
            trans_conv1.append(Conv2DTranspose(filters=16 * 2 ** filter,
                                               kernel_size=2, strides=2,
                                               padding='same')(res_bottle[i]))
            res_up1.append(ResBlock(16 * 2 ** filter,
                                    kernel_size=kernel_size,
                                    activation=activation,
                                    initializer=initializer)(trans_conv1[i]))
            bridge1.append(ConvBridgeBlock(
                16 * 2 ** filter, kernel_size=kernel_size,
                activation=activation,
                initializer=initializer)(res_down2[i]))
            concat1.append(concatenate([res_up1[i], bridge1[i]]))

            trans_conv2.append(Conv2DTranspose(filters=8 * 2 ** filter,
                                               kernel_size=2,
                                               strides=2,
                                               padding='same')(concat1[i]))
            res_up2.append(ResBlock(8 * 2 ** filter,
                                    kernel_size=kernel_size,
                                    activation=activation,
                                    initializer=initializer)(trans_conv2[i]))
            bridge2.append(ConvBridgeBlock(
                8 * 2 ** filter, kernel_size=kernel_size,
                activation=activation,
                initializer=initializer)(res_down1[i]))
            concat2.append(concatenate([res_up2[i], bridge2[i]]))

            # Output
            output_layer.append(Conv2D(self.output_channels, 1, strides=1,
                                       padding='same',
                                       activation="sigmoid")(concat2[i]))

            # input for next w-net
            input.append(concatenate([input[i], output_layer[i]]))

        model = tf.keras.Model(inputs=input[0], outputs=output_layer[i],
                               name="Hyper-UNetResnet")

        model.compile(
            optimizer=opt, loss=loss,
            metrics=['accuracy', tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall"),
                     tf.keras.metrics.AUC(name="auc", curve="PR"),
                     ]
        )

        return model
# *===========================================================================*
