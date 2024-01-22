# W-Net structure copied from the paper:
# The Little W-Net That Could: State-of-the-Art Retinal Vessel Segmentation
# with Minimalistic Models by Galdran et al.
# Link: https://arxiv.org/abs/2009.01907

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Conv2DTranspose,
                                     Input, BatchNormalization, Concatenate)

from src.pipelines.tensorflow_v2.helpers.train_test import _TfPnsMixin
from src.pipelines.tensorflow_v2.models.unet_resnet import ResBlock


# *============================ Conv Bridge Block ============================*
class ConvBridgeBlock(tf.keras.Model):
    """
    A Convolutional Bridge Block to be used in a W-Net
    """
    def __init__(self,
                 channels: int,
                 activation: str,
                 initializer: str):
        """
        Instantiates a ConvBridgeBlock object

        :param channels: number of filters required for the conv layers
        :param activation: the activation function to use
        :param initializer: the weight initializer to use
        """
        super().__init__()

        self.conv = Conv2D(filters=channels, kernel_size=3, strides=1,
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


# *=============================== Mini U-Net ================================*
class MiniUnet(tf.keras.Model):
    """
    A mini U-Net, two of these are joined to make a W-Net model
    """
    def __init__(self,
                 output_channels: int,
                 activation: str = "relu",
                 initializer: str = "he_normal",
                 filters: int = 0):
        """
        Instantiates a MiniUnet object.

        :param output_channels: the number of output channels to use in the
         final layer; this is related to the number of classes in the final
         prediction
        :param activation: the activation to use in all blocks apart from
         the last
        :param initializer: the initialiser to use in all blocks apart from
         the last
        :param filters: the filter multiple to use
        """
        super().__init__()
        if initializer == "he_normal":
            initializer = tf.keras.initializers.he_normal(seed=3141)
        else:
            initializer = tf.keras.initializers.glorot_uniform(seed=3141)
        # Components

        # Contracting
        self.res_down1 = ResBlock(8 * 2 ** filters, decode=True,
                                  activation=activation,
                                  initializer=initializer)
        self.pool1 = MaxPool2D(pool_size=2, strides=2)

        self.res_down2 = ResBlock(16 * 2 ** filters, decode=True,
                                  activation=activation,
                                  initializer=initializer)
        self.pool2 = MaxPool2D(pool_size=2, strides=2)

        self.res_bottle = ResBlock(32 * 2 ** filters, decode=True,
                                   activation=activation,
                                   initializer=initializer)

        # Expanding
        self.trans_conv1 = Conv2DTranspose(filters=16 * 2 ** filters,
                                           kernel_size=2, strides=2,
                                           padding='same',
                                           activation=activation,
                                           kernel_initializer=initializer)
        self.res_up1 = ResBlock(16 * 2 ** filters, activation=activation,
                                initializer=initializer)
        self.bridge1 = ConvBridgeBlock(16 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.concat1 = Concatenate()

        self.trans_conv2 = Conv2DTranspose(filters=8 * 2 ** filters,
                                           kernel_size=2,
                                           activation=activation,
                                           strides=2, padding='same',
                                           kernel_initializer=initializer)
        self.res_up2 = ResBlock(8 * 2 ** filters, activation=activation,
                                initializer=initializer)
        self.bridge2 = ConvBridgeBlock(8 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.concat2 = Concatenate()

        # Output
        self.output_layer = Conv2D(output_channels, 1, strides=1,
                                   padding='same',
                                   activation="sigmoid",
                                   name="classification_layer")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
         Applies a mini U-Net to the input.

         :param inputs: an input image
         :return: the output of a mini U-Net
         """
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

    def model(self,
              shape: Tuple[int, int, int] = (512, 512, 1)) -> tf.keras.Model:
        """
        Returns a U-Net model as tf.keras.Model. This is a workaround to use
        the functional api, which allows the model to be viewed.

        :param shape: the shape of the input
        :return: the tf.keras.Model instantiated using the functional api
        """
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def print_all_layers(self) -> None:
        """
        Prints all the layers in the model, including the layers in
        the subclasses which make up the model. This uses the model()
        workaround function.

        :return: None
        """
        model_layers = self.model().layers

        for layer in model_layers:
            try:
                for l in layer.layers:
                    print(l)
            except:
                print(layer)

    def get_all_layers(self) -> List[tf.keras.layers.Layer]:
        """
        Returns all the layers in the model, including the layers in
        the subclasses which make up the model. This uses the model()
        workaround function.

        :return: a list of layers
        """
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
    A W-Net model. This model combines two Mini U-Nets where the prediction of
    the first Mini U-Net is concatenated to the first
    """

    def __init__(self,
                 output_channels: int,
                 activation: str = "relu",
                 initializer: str = "he_normal",
                 filters: int = 0):
        """
        Instantiates a WNet object.

        :param output_channels: the number of output channels to use in the
         final layer; this is related to the number of classes in the final
         prediction
        :param activation: the activation to use in all blocks apart from
         the last
        :param initializer: the initialiser to use in all blocks apart from
         the last
        :param filters: the filter multiple to use
        """

        super().__init__()

        self.activation = tf.nn.relu

        # Channels set for binary prediction
        self.unet1 = MiniUnet(output_channels, activation, initializer,
                              filters)
        self.unet2 = MiniUnet(output_channels, activation, initializer,
                              filters)
        self.concat = Concatenate()

    def call(self,
             input_tensor: tf.Tensor,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies a W-Net model to an input image, which is a call of two
        sequent mini U-Nets. If the W-Net is not training then only the
        output of the second mini U-Net is returned.

        :param input_tensor: an input image
        :param training: whether the W-Net is being applied to a training
         sample
        :return: either the ouput of both mini U-Nets or the output
         of the second mini U-Net
        """
        x1 = self.unet1(input_tensor)
        x = self.concat([input_tensor, x1])
        x2 = self.unet2(x)

        if not training:
            return x2

        return x1, x2

    def model(self,
              shape: Tuple[int, int, int] = (512, 512, 1)) -> tf.keras.Model:
        """
        Returns a U-Net model as tf.keras.Model. This is a workaround to use
        the functional api, which allows the model to be viewed.

        :param shape: the shape of the input
        :return: the tf.keras.Model instantiated using the functional api
        """

        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def print_all_layers(self) -> None:
        """
        Prints all the layers in the model, including the layers in
        the subclasses which make up the model. This uses the model()
        workaround function.

        :return: None
        """
        model_layers = self.model().layers

        for layer in model_layers:
            try:
                for l in layer.layers:
                    print(l)
            except:
                print(layer)

    def get_all_layers(self) -> List[tf.keras.layers.Layer]:
        """
        Returns all the layers in the model, including the layers in
        the subclasses which make up the model. This uses the model()
        workaround function.

        :return: a list of layers
        """
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
