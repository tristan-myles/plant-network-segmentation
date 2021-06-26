import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose

from src.pipelines.tensorflow_v2.helpers.train_test import _TfPnsMixin


# *========================== U-Net Building Block ===========================*
class UnetBlock(tf.keras.Model):
    def __init__(self, num_filters: int, kernel_size: int,
                 decode: bool = False, encode: bool = False,
                 batch_norm: bool = False, padding: str = "same",
                 activation="relu", initializer='he_normal',
                 name: str = ""):
        """
        A building block to be used when constructing a U-Net model

        Three possible modes for this block:
        1) Encode, i.e. contracting path | requires encode=True
        2) Decode, i.e. expanding path | requires decode=True
        3) Bottleneck | requires encode=False and decode=False

        .. note::  Glorot uniform is default kernel initialisation

        :param num_filters: number of filters required for the conv layers
        :param kernel_size: h,w of the kernel
        :param decode: whether the block will be used in the expanding path
        :param encode: whether the block will be used in the contracting path
        :param padding: how to pad the input
        :param activation: the activation function to use
        :param initializer: the weigh initializer to use
        :param batch_norm: whether to include batch normalization layers
        :param name: the name of the block
        """

        super().__init__(name=name)

        self.conv1 = Conv2D(filters=num_filters, kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            kernel_initializer=initializer)
        self.conv2 = Conv2D(filters=num_filters, kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            kernel_initializer=initializer)
        if batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()

        if decode:
            # half the x,y dimensions each pool step
            self.pool = MaxPool2D(pool_size=(2, 2), strides=2)
        elif encode:
            # default axis is -1 => the filter axis
            self.concat = tf.keras.layers.Concatenate()

            # double the x,y dimensions each pool step, notice no activation
            self.conv_trans = Conv2DTranspose(filters=num_filters,
                                              kernel_size=2, strides=2,
                                              padding=padding,
                                              activation=activation,
                                              kernel_initializer=initializer)
        elif decode and encode:
            raise ValueError("Decode and Encode can't both be True")

        # Store user-defined input to be used in call step
        self.batch_norm = batch_norm
        self.decode = decode
        self.encode = encode

    def downblock(self, input_tensor):
        x = self.conv1(input_tensor)

        if self.batch_norm:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.batch_norm:
            x = self.bn2(x)

        pool_out = self.pool(x)

        return x, pool_out

    def upblock(self, input_tensor, downblock_output):
        x = self.conv_trans(input_tensor)
        x = self.concat([x, downblock_output])

        x = self.conv1(x)

        if self.batch_norm:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.batch_norm:
            x = self.bn2(x)

        return x

    def bottleneck(self, input_tensor):
        x = self.conv1(input_tensor)

        if self.batch_norm:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.batch_norm:
            x = self.bn2(x)

        return x

    def call(self, input_tensor: tf.Tensor,
             downblock_output: tf.Tensor = None):
        if self.decode:
            return self.downblock(input_tensor)

        elif self.encode:
            if downblock_output is None:
                raise ValueError("Please provide output to use for the skip "
                                 "connection")

            return self.upblock(input_tensor, downblock_output)

        else:
            return self.bottleneck(input_tensor)


# *================================== U-Net ==================================*
class Unet(tf.keras.Model, _TfPnsMixin):
    # Olaf Ronneberger et al. U-Net
    def __init__(self, output_channels, activation="relu",
                 initializer="he_normal", filters=3):
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
        # Layer 1
        self.conv_down1 = UnetBlock(num_filters=8 * 2**filters, kernel_size=3,
                                    decode=True, activation=activation,
                                    initializer=initializer, name="down1")

        # Layer 2
        self.conv_down2 = UnetBlock(num_filters=16 * 2**filters, kernel_size=3,
                                    decode=True,  activation=activation,
                                    initializer=initializer, name="down2")

        # Layer 3
        self.conv_down3 = UnetBlock(num_filters=32 * 2**filters, kernel_size=3,
                                    decode=True,  activation=activation,
                                    initializer=initializer, name="down3")

        # Layer 4
        self.conv_down4 = UnetBlock(num_filters=64 * 2**filters, kernel_size=3,
                                    decode=True,  activation=activation,
                                    initializer=initializer, name="down4")

        # Bottleneck
        self.conv_bottle = UnetBlock(num_filters=128 * 2**filters,
                                     kernel_size=3,  activation=activation,
                                     initializer=initializer,
                                     name="bottleneck")

        # Expanding

        # Layer 1
        # No activation ... Since skip happens before the activation
        self.conv_up1 = UnetBlock(num_filters=64 * 2**filters, kernel_size=3,
                                  encode=True,  activation=activation,
                                  initializer=initializer,
                                  name="up1")
        # Layer 2
        self.conv_up2 = UnetBlock(num_filters=32 * 2**filters, kernel_size=3,
                                  encode=True,  activation=activation,
                                  initializer=initializer,
                                  name="up2")
        # Layer 3
        self.conv_up3 = UnetBlock(num_filters=16 * 2**filters, kernel_size=3,
                                  encode=True,  activation=activation,
                                  initializer=initializer,
                                  name="up3")
        # Layer 4
        self.conv_up4 = UnetBlock(num_filters=8 * 2**filters, kernel_size=3,
                                  encode=True,  activation=activation,
                                  initializer=initializer, name="up4")

        # Output
        self.output_layer = Conv2D(output_channels, 1, strides=1,
                                   padding='same',
                                   activation="sigmoid",
                                   name="classification_layer")  # 64x64 -> 128x128

    def call(self, inputs):
        # Contracting
        #  Layer 1
        down1, x = self.conv_down1(inputs)

        #  Layer 2
        down2, x = self.conv_down2(x)

        #  Layer 3
        down3, x = self.conv_down3(x)

        #  Layer 4
        down4, x = self.conv_down4(x)

        #  Bottleneck
        x = self.conv_bottle(x)

        # Expanding
        # Layer 1
        x = self.conv_up1(x, down4)

        # Layer 2
        x = self.conv_up2(x, down3)

        # Layer 3
        x = self.conv_up3(x, down2)

        # Layer 4
        x = self.conv_up4(x, down1)

        x = self.output_layer(x)

        return x

    def model(self, shape=(512, 512, 1)):
        x = Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def print_all_layers(self):
        model_layers = self.model().layers

        print(model_layers[0])

        for layer in model_layers[1:-1]:
            for l in layer.layers:
                print(l)

        print(model_layers[-1])

    def get_all_layers(self):
        model_layers = self.model().layers
        layers = []
        #         layers.append(model_layers[0])

        for layer in model_layers[1:-1]:
            for l in layer.layers:
                layers.append(l)

        layers.append(model_layers[-1])

        return layers
# *===========================================================================*
