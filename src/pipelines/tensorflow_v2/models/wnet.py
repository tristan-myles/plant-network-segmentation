# W-Net structure copied from the paper:
# The Little W-Net That Could: State-of-the-Art Retinal Vessel Segmentation
# with Minimalistic Models by Galdran et al.
# Link: https://arxiv.org/abs/2009.01907

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Conv2DTranspose,
                                     Input, BatchNormalization, Concatenate)

from src.pipelines.tensorflow_v2.models.unet_resnet import ResBlock
from src.pipelines.tensorflow_v2.helpers.train_test import _TfPnsMixin


# *============================ Conv Bridge Block ============================*
class ConvBridgeBlock(tf.keras.Model):
    def __init__(self, channels, activation, initializer):
        super().__init__()

        self.conv = Conv2D(filters=channels, kernel_size=3, strides=1,
                           padding='same',
                           kernel_initializer=initializer)
        self.bn = BatchNormalization()
        self.activation = activation

    def call(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x1)
        x1 = self.activation(x1)

        return x1


# *=============================== Mini U-Net ================================*
class MiniUnet(tf.keras.Model):
    # Olaf Ronneberger et al. U-Net
    def __init__(self, output_channels, activation="relu",
                 initializer="he_normal", filters=0):
        super().__init__()
        if initializer == "he_normal":
            initializer = tf.keras.initializers.he_normal(seed=3141)
        else:
            initializer = tf.keras.initializers.glorot_uniform(seed=3141)
        # Components

        # Contracting
        self.res_down1 = ResBlock(8 * 2**filters, decode=True,
                                  activation=activation,
                                  initializer=initializer)
        self.pool1 = MaxPool2D(pool_size=2, strides=2)

        self.res_down2 = ResBlock(16 * 2**filters, decode=True,
                                  activation=activation,
                                  initializer=initializer)
        self.pool2 = MaxPool2D(pool_size=2, strides=2)

        self.res_bottle = ResBlock(32 * 2**filters, decode=True,
                                   activation=activation,
                                   initializer=initializer)

        # Expanding
        self.trans_conv1 = Conv2DTranspose(filters=16 * 2**filters,
                                           kernel_size=2, strides=2,
                                           padding='same',
                                           activation=activation,
                                           kernel_initializer=initializer)
        self.res_up1 = ResBlock(16 * 2**filters, activation=activation,
                                initializer=initializer)
        self.bridge1 = ConvBridgeBlock(16 * 2**filters, activation=activation,
                                       initializer=initializer)
        self.concat1 = Concatenate()

        self.trans_conv2 = Conv2DTranspose(filters=8 * 2**filters,
                                           kernel_size=2,
                                           activation=activation,
                                           strides=2, padding='same',
                                           kernel_initializer=initializer)
        self.res_up2 = ResBlock(8 * 2**filters, activation=activation,
                                initializer=initializer)
        self.bridge2 = ConvBridgeBlock(8 * 2**filters,
                                       activation=activation,
                                       initializer=initializer)
        self.concat2 = Concatenate()

        # Output
        self.output_layer = Conv2D(output_channels, 1, strides=1,
                                   padding='same',
                                   activation="sigmoid",
                                   name="classification_layer")

    def call(self, inputs):
        # Contracting
        down1 = self.res_down1(inputs)
        x = self.pool1(down1)

        down2 = self.res_down2(x)
        x = self.pool2(down2)

        # Bottle
        x = self.res_bottle(x)

        # Expanding
        x = self.trans_conv1(x)
        x = self.res_up1(x)

        # Why?
        down2 = self.bridge1(down2)
        x = self.concat1([x, down2])

        x = self.trans_conv2(x)
        x = self.res_up2(x)
        down1 = self.bridge2(down1)
        x = self.concat2([x, down1])

        x = self.output_layer(x)

        return x

    def model(self, shape=(512, 512, 1)):
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def print_all_layers(self):
        model_layers = self.model().layers

        for layer in model_layers:
            try:
                for l in layer.layers:
                    print(l)
            except:
                print(layer)

    def get_all_layers(self):
        model_layers = self.model().layers
        layers = []
        for layer in model_layers:
            try:
                for l in layer.layers:
                    layers.append(l)
            except:
                layers.append(model_layers[-1])

        return layers


# *================================== W-Net ==================================*
class WNet(tf.keras.Model, _TfPnsMixin):
    """
    Combines two Mini U-Nets where the prediction of the first Mini U-Net is
    concatenated to the first
    """
    def __init__(self, output_channels, activation="relu",
                 initializer="he_normal", filters=0):
        super().__init__()

        if activation == "relu":
            activation = tf.nn.relu
        else:
            activation = tf.nn.selu
        self.activation = activation

        # Channels set for binary prediction
        self.unet1 = MiniUnet(output_channels, activation, initializer,
                              filters)
        self.unet2 = MiniUnet(output_channels, activation, initializer,
                              filters)
        self.concat = Concatenate()

    def call(self, input_tensor, training=True):
        x1 = self.unet1(input_tensor)
        x = self.concat([input_tensor, x1])
        x2 = self.unet2(x)

        if not training:
            return x2

        return x1, x2

    def model(self, shape=(512, 512, 1)):
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def print_all_layers(self):
        model_layers = self.model().layers

        for layer in model_layers:
            try:
                for l in layer.layers:
                    print(l)
            except:
                print(layer)

    def get_all_layers(self):
        model_layers = self.model().layers
        layers = []
        for layer in model_layers:
            try:
                for l in layer.layers:
                    layers.append(l)
            except:
                layers.append(model_layers[-1])

        return layers
# *===========================================================================*
