# Model structure adapted from: Road Extraction by Deep Residual U-Net by
# Zhang, Liu, and Wang
# Link: https://arxiv.org/abs/1711.10684
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Concatenate,
                                     Conv2DTranspose, BatchNormalization,
                                     Input)


# *============================= Residual Block ==============================*
class ResBlock(tf.keras.Model):
    def __init__(self, channels, stride=1, decode=False):
        super().__init__()

        he_initializer = tf.keras.initializers.he_normal(seed=3141)

        # for dotted shortcut, i.e stride = 2
        self.flag = (stride != 1)

        # for concatenated feature map
        self.decode = decode

        # res block
        self.conv1 = Conv2D(filters=channels, kernel_size=3, strides=stride,
                            padding='same',
                            kernel_initializer=he_initializer)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=channels, kernel_size=3, strides=1,
                            padding='same',
                            kernel_initializer=he_initializer)
        self.bn2 = BatchNormalization()

        if self.decode or self.flag:
            # 1x1 convolution
            self.conv3 = Conv2D(channels, 1, stride)
            self.bn3 = BatchNormalization()

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = tf.nn.relu(x1)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1)

        # Matching dims (i.e. projection shortcut)
        if self.flag or self.decode:
            x = self.conv3(x)
            x = self.bn3(x)

        x1 += x

        # Addition is before the activation...
        x1 = tf.nn.relu(x1)

        return x1


# *============================== ResNet U-Net ===============================*
class UnetResnet(tf.keras.Model):
    def __init__(self, output_channels):
        super().__init__()
        he_initializer = tf.keras.initializers.he_normal(seed=3141)

        # Contracting
        self.down_conv1 = Conv2D(filters=64, kernel_size=7, strides=2,
                                 padding='same',
                                 kernel_initializer=he_initializer)
        self.down_bn = BatchNormalization()
        # Changed strides to pool size to 2
        self.mp1 = MaxPool2D(pool_size=2, strides=2)

        self.down_block_2_1 = ResBlock(64)
        self.down_block_2_2 = ResBlock(64)
        self.down_block_2_3 = ResBlock(64)

        self.down_block_3_1 = ResBlock(128, 2)
        self.down_block_3_2 = ResBlock(128)
        self.down_block_3_3 = ResBlock(128)
        self.down_block_3_4 = ResBlock(128)

        self.down_block_4_1 = ResBlock(256, 2)
        self.down_block_4_2 = ResBlock(256)
        self.down_block_4_3 = ResBlock(256)
        self.down_block_4_4 = ResBlock(256)
        self.down_block_4_5 = ResBlock(256)
        self.down_block_4_6 = ResBlock(256)

        self.down_block_5_1 = ResBlock(512, 2)
        self.down_block_5_2 = ResBlock(512)
        self.down_block_5_3 = ResBlock(512)

        self.conv_up1 = Conv2DTranspose(filters=256, kernel_size=1, strides=2,
                                        padding='valid',
                                        kernel_initializer=he_initializer)
        # default axis is -1 => the filter axis
        self.conv_up_concat_1 = Concatenate()

        self.up_block_1_1 = ResBlock(256, decode=True)
        self.up_block_1_2 = ResBlock(256)
        self.up_block_1_3 = ResBlock(256)
        self.up_block_1_4 = ResBlock(256)
        self.up_block_1_5 = ResBlock(256)

        # Layer 2
        self.conv_up2 = Conv2DTranspose(filters=128, kernel_size=1, strides=2,
                                        padding='valid',
                                        kernel_initializer=he_initializer)
        self.conv_up_concat_2 = Concatenate()

        self.up_block_2_1 = ResBlock(128, decode=True)
        self.up_block_2_2 = ResBlock(128)
        self.up_block_2_3 = ResBlock(128)

        # Layer 3
        self.conv_up3 = Conv2DTranspose(filters=64, kernel_size=1, strides=2,
                                        padding='same',
                                        kernel_initializer=he_initializer)
        self.conv_up_concat_3 = Concatenate()

        self.up_block_3_1 = ResBlock(64, decode=True)
        self.up_block_3_2 = ResBlock(64)
        self.up_block_3_3 = ResBlock(64)

        # Layer 4
        self.conv_up4 = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                                        padding='same',
                                        kernel_initializer=he_initializer)

        self.conv_up_concat_4 = Concatenate()
        self.up_conv4 = Conv2D(filters=64, kernel_size=3, strides=2,
                               padding='same',
                               kernel_initializer=he_initializer)
        self.up_bn = BatchNormalization()

        # Think about whether this needs to be activated
        self.conv_up5 = Conv2DTranspose(filters=64, kernel_size=7, strides=2,
                                        padding='same',
                                        kernel_initializer=he_initializer)
        self.output_layer = Conv2D(output_channels, 1, strides=1,
                                   padding='same',
                                   activation="sigmoid")  # 64x64 -> 128x128

    def call(self, x):
        x = self.down_conv1(x)
        x = self.down_bn(x)
        down_conv1 = tf.nn.relu(x)
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

        x = self.conv_up2(x)
        x = self.conv_up_concat_2([x, down_3_4])
        x = self.up_block_2_1(x)
        x = self.up_block_2_2(x)
        x = self.up_block_2_3(x)

        x = self.conv_up3(x)
        x = self.conv_up_concat_3([x, down_2_3])
        x = self.up_block_3_1(x)
        x = self.up_block_3_2(x)
        x = self.up_block_3_3(x)

        x = self.conv_up4(x)
        x = self.conv_up_concat_4([x, down_conv1])
        x = self.up_bn(x)
        x = tf.nn.relu(x)

        # Think about this.....
        x = self.conv_up5(x)

        x = self.output_layer(x)

        return x

    def model(self, shape=(288, 288, 1)):
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
