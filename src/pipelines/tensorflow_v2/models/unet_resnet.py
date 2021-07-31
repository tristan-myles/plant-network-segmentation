# Model structure adapted from: Road Extraction by Deep Residual U-Net by
# Zhang, Liu, and Wang
# Link: https://arxiv.org/abs/1711.10684
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Concatenate,
                                     Conv2DTranspose, BatchNormalization,
                                     Input)

from src.pipelines.tensorflow_v2.helpers.train_test import _TfPnsMixin


# *============================= Residual Block ==============================*
class ResBlock(tf.keras.Model):
    """
    A Residual block
    """

    def __init__(self,
                 channels,
                 initializer,
                 activation,
                 stride=1,
                 decode=False):
        """
        Instantiates a ResBlock object.

        :param channels: number of filters required for the conv layers
        :param activation: the activation function to use
        :param initializer: the weight initializer to use
        :param stride: the stride length
        :param decode: whether this ResBlock is being used in expansive path of
         the U-Net
        """
        super().__init__()
        self.activation = activation
        # for dotted shortcut, i.e stride = 2
        self.flag = (stride != 1)

        # for concatenated feature map
        self.decode = decode

        # res block
        self.conv1 = Conv2D(filters=channels, kernel_size=3, strides=stride,
                            padding='same',
                            kernel_initializer=initializer)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=channels, kernel_size=3, strides=1,
                            padding='same',
                            kernel_initializer=initializer)
        self.bn2 = BatchNormalization()

        if self.decode or self.flag:
            # 1x1 convolution
            # using for input that's been activated already
            self.conv3 = Conv2D(channels, 1, stride,
                                kernel_initializer=initializer)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies a ResBlock to the an input

        :param x: the input to apply the ResBlock to
        :return: the output of the ResBlock
        """
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x1 = self.bn1(x1)

        # apply activation before bn - differs from BN paper
        x1 = self.conv2(x1)
        x1 = self.activation(x1)

        # Matching dims (i.e. projection shortcut)
        # No need to activate
        if self.flag or self.decode:
            x = self.conv3(x)
            x = self.activation(x)

        x1 += x
        x1 = self.bn2(x1)

        return x1


# *============================== ResNet U-Net ===============================*
class UnetResnet(tf.keras.Model, _TfPnsMixin):
    """
    A U-Net model, which has a ResNet 34 backbone
    """

    def __init__(self,
                 output_channels: int,
                 activation: str = "relu",
                 initializer: str = "he_normal",
                 filters: int = 3):
        """
        Instantiates a U-Net ResNet 34 model

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

        if activation == "relu":
            activation = tf.nn.relu
        else:
            activation = tf.nn.selu
        self.activation = activation

        # Contracting
        # valid padding since down sampling
        self.down_conv1 = Conv2D(filters=8 * 2 ** filters, kernel_size=7,
                                 strides=2,
                                 padding='same',
                                 kernel_initializer=initializer)
        self.down_bn = BatchNormalization()
        # Changed strides to pool size to 2
        self.mp1 = MaxPool2D(pool_size=2, strides=2)

        self.down_block_2_1 = ResBlock(8 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_2_2 = ResBlock(8 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_2_3 = ResBlock(8 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)

        self.down_block_3_1 = ResBlock(16 * 2 ** filters, stride=2,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_3_2 = ResBlock(16 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_3_3 = ResBlock(16 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_3_4 = ResBlock(16 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)

        self.down_block_4_1 = ResBlock(32 * 2 ** filters, stride=2,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_4_2 = ResBlock(32 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_4_3 = ResBlock(32 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_4_4 = ResBlock(32 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_4_5 = ResBlock(32 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_4_6 = ResBlock(32 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)

        self.down_block_5_1 = ResBlock(64 * 2 ** filters, stride=2,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_5_2 = ResBlock(64 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)
        self.down_block_5_3 = ResBlock(64 * 2 ** filters,
                                       activation=activation,
                                       initializer=initializer)

        self.conv_up1 = Conv2DTranspose(filters=32 * 2 ** filters,
                                        kernel_size=2,
                                        strides=2, activation=activation,
                                        kernel_initializer=initializer)
        # default axis is -1 => the filter axis
        self.conv_up_concat_1 = Concatenate()

        self.up_block_1_1 = ResBlock(32 * 2 ** filters, activation=activation,
                                     initializer=initializer, decode=True)
        self.up_block_1_2 = ResBlock(32 * 2 ** filters, activation=activation,
                                     initializer=initializer)
        self.up_block_1_3 = ResBlock(32 * 2 ** filters, activation=activation,
                                     initializer=initializer)
        self.up_block_1_4 = ResBlock(32 * 2 ** filters, activation=activation,
                                     initializer=initializer)
        self.up_block_1_5 = ResBlock(32 * 2 ** filters, activation=activation,
                                     initializer=initializer)
        self.up_block_1_6 = ResBlock(32 * 2 ** filters, activation=activation,
                                     initializer=initializer)

        # Layer 2
        self.conv_up2 = Conv2DTranspose(filters=16 * 2 ** filters,
                                        kernel_size=2,
                                        strides=2, activation=activation,
                                        kernel_initializer=initializer)
        self.conv_up_concat_2 = Concatenate()

        self.up_block_2_1 = ResBlock(16 * 2 ** filters, activation=activation,
                                     initializer=initializer, decode=True)
        self.up_block_2_2 = ResBlock(16 * 2 ** filters, activation=activation,
                                     initializer=initializer)
        self.up_block_2_3 = ResBlock(16 * 2 ** filters, activation=activation,
                                     initializer=initializer)
        self.up_block_2_4 = ResBlock(16 * 2 ** filters, activation=activation,
                                     initializer=initializer)

        # Layer 3
        self.conv_up3 = Conv2DTranspose(filters=8 * 2 ** filters,
                                        kernel_size=2,
                                        strides=2, activation=activation,
                                        kernel_initializer=initializer)
        self.conv_up_concat_3 = Concatenate()

        self.up_block_3_1 = ResBlock(8 * 2 ** filters, activation=activation,
                                     initializer=initializer, decode=True)
        self.up_block_3_2 = ResBlock(8 * 2 ** filters, activation=activation,
                                     initializer=initializer)
        self.up_block_3_3 = ResBlock(8 * 2 ** filters, activation=activation,
                                     initializer=initializer)

        # Layer 4
        self.conv_up4 = Conv2DTranspose(filters=8 * 2 ** filters,
                                        kernel_size=2,
                                        strides=2, activation=activation,
                                        kernel_initializer=initializer)

        self.conv_up_concat_4 = Concatenate()
        # Activation corresponding to first layer
        self.up_conv4 = Conv2D(filters=8 * 2 ** filters, kernel_size=7,
                               strides=1, padding='same',
                               kernel_initializer=initializer)
        self.up_bn = BatchNormalization()

        # Think about whether this needs to be activated: No since activation
        # corresponding to this first layer is dealt with above
        self.conv_up5 = Conv2DTranspose(filters=8 * 2 ** filters,
                                        kernel_size=2,
                                        strides=2, activation=activation,
                                        kernel_initializer=initializer)
        self.output_layer = Conv2D(output_channels, 1, strides=1,
                                   padding='same',
                                   activation="sigmoid")  # 64x64 -> 128x128

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies a U-Net ResNet34 model to the input.

        :param inputs: an input image
        :return: the output of a U-Net ResNet34 model
        """
        x = self.down_conv1(x)
        x = self.activation(x)
        down_conv1 = self.down_bn(x)
        x = self.mp1(down_conv1)

        x = self.down_block_2_1(x)
        x = self.down_block_2_2(x)
        down_2_3 = self.down_block_2_3(x)

        x = self.down_block_3_1(down_2_3)
        x = self.down_block_3_2(x)
        x = self.down_block_3_3(x)
        down_3_4 = self.down_block_3_4(x)

        x = self.down_block_4_1(down_3_4)
        x = self.down_block_4_2(x)
        x = self.down_block_4_3(x)
        x = self.down_block_4_4(x)
        x = self.down_block_4_5(x)
        down_4_6 = self.down_block_4_6(x)

        x = self.down_block_5_1(down_4_6)
        x = self.down_block_5_2(x)
        x = self.down_block_5_3(x)

        # Expanding
        x = self.conv_up1(x)
        x = self.conv_up_concat_1([x, down_4_6])
        x = self.up_block_1_1(x)
        x = self.up_block_1_2(x)
        x = self.up_block_1_3(x)
        x = self.up_block_1_4(x)
        x = self.up_block_1_5(x)
        x = self.up_block_1_6(x)

        x = self.conv_up2(x)
        x = self.conv_up_concat_2([x, down_3_4])
        x = self.up_block_2_1(x)
        x = self.up_block_2_2(x)
        x = self.up_block_2_3(x)
        x = self.up_block_2_4(x)

        x = self.conv_up3(x)
        x = self.conv_up_concat_3([x, down_2_3])
        x = self.up_block_3_1(x)
        x = self.up_block_3_2(x)
        x = self.up_block_3_3(x)

        x = self.conv_up4(x)
        x = self.conv_up_concat_4([x, down_conv1])
        x = self.up_conv4(x)
        x = self.activation(x)
        x = self.up_bn(x)

        # For each layer with a stride of 2 there will be both an activation
        # and a transposed convolution (with no activation)
        x = self.conv_up5(x)

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

        print(model_layers[0])

        for layer in model_layers[1:-1]:
            for l in layer.layers:
                print(l)

        print(model_layers[-1])

    def get_all_layers(self) -> List[tf.keras.layers.Layer]:
        """
        Returns all the layers in the model, including the layers in
        the subclasses which make up the model. This uses the model()
        workaround function.

        :return: a list of layers
        """
        model_layers = self.model().layers
        layers = []
        #         layers.append(model_layers[0])

        for layer in model_layers[1:-1]:
            for l in layer.layers:
                layers.append(l)

        layers.append(model_layers[-1])

        return layers
# *===========================================================================*
