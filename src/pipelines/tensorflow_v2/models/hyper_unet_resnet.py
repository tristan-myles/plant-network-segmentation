from kerastuner import HyperModel
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D,
                                     Conv2DTranspose, concatenate,
                                     BatchNormalization)
from src.pipelines.tensorflow_v2.losses.custom_losses import *


# *============================= Residual Block ==============================*
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, activation, initializer,
                 stride=1, decode=False):
        super().__init__()
        self.activation = activation

        # for dotted shortcut, i.e stride = 2
        self.flag = (stride != 1)

        # for concatenated feature map
        self.decode = decode

        # res block
        self.conv1 = Conv2D(filters=channels, kernel_size=kernel_size,
                            strides=stride,
                            padding='same',
                            kernel_initializer=initializer)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=channels, kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            kernel_initializer=initializer)
        self.bn2 = BatchNormalization()

        if self.decode or self.flag:
            # 1x1 convolution
            # using for input that's been activated already
            self.conv3 = Conv2D(channels, 1, stride,
                                kernel_initializer=initializer)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x1 = self.bn1(x1)

        x1 = self.conv2(x1)
        x1 = self.activation(x1)

        # Matching dims (i.e. projection shortcut)
        if self.flag or self.decode:
            x = self.conv3(x)
            x = self.activation(x)

        x1 += x
        x1 = self.bn2(x1)

        # Addition is before the activation...

        return x1


class HyperUnetResnet(HyperModel):
    def __init__(self, input_shape, output_channels):
        self.input_shape = input_shape
        self.output_channels = output_channels

    def build(self, hp):
        # Search space
        # create the search space
        initializer = hp.Choice("initializer", ["he_normal", "glorot_uniform"])
        activation = hp.Choice("activation", ["relu", "selu"])
        filter = hp.Choice("filters", [0, 1, 2, 3])
        kernel_size = hp.Choice("kernel_size", [3])
        optimizer = hp.Choice("optimizer", ["adam", "sgd"])
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

        # Contracting
        # valid padding since down sampling
        down_conv1 = Conv2D(filters=8 * 2**filter, kernel_size=7, strides=2,
                            padding='same')(input)
        down1_activated = activation(down_conv1)
        down1_bn = BatchNormalization()(down1_activated)
        # Changed strides to pool size to 2
        mp1 = MaxPool2D(pool_size=2, strides=2)(down1_bn)

        down_block_2_1 = ResBlock(channels=8 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(mp1)
        down_block_2_2 = ResBlock(channels=8 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_2_1)
        down_block_2_3 = ResBlock(channels=8 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_2_2)

        down_block_3_1 = ResBlock(channels=16 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer,
                                  stride=2)(down_block_2_3)
        down_block_3_2 = ResBlock(channels=16 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_3_1)
        down_block_3_3 = ResBlock(channels=16 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_3_2)
        down_block_3_4 = ResBlock(channels=16 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_3_3)

        down_block_4_1 = ResBlock(channels=32 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer,
                                  stride=2)(down_block_3_4)
        down_block_4_2 = ResBlock(channels=32 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_4_1)
        down_block_4_3 = ResBlock(channels=32 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_4_2)
        down_block_4_4 = ResBlock(channels=32 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_4_3)
        down_block_4_5 = ResBlock(channels=32 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_4_4)
        down_block_4_6 = ResBlock(channels=32 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_4_5)

        down_block_5_1 = ResBlock(channels=64 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer,
                                  stride=2)(down_block_4_6)
        down_block_5_2 = ResBlock(channels=64 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_5_1)
        down_block_5_3 = ResBlock(channels=64 * 2**filter,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  initializer=initializer)(down_block_5_2)

        conv_up1 = Conv2DTranspose(filters=32 * 2**filter, kernel_size=2,
                                   strides=2,
                                   kernel_initializer=initializer)(
            down_block_5_3)
        # default axis is -1 => the filter axis
        conv_up_concat_1 = concatenate([conv_up1, down_block_4_6])

        up_block_1_1 = ResBlock(channels=32 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer,
                                decode=True)(conv_up_concat_1)
        up_block_1_2 = ResBlock(channels=32 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_1_1)
        up_block_1_3 = ResBlock(channels=32 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_1_2)
        up_block_1_4 = ResBlock(channels=32 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_1_3)
        up_block_1_5 = ResBlock(channels=32 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_1_4)
        up_block_1_6 = ResBlock(channels=32 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_1_5)

        # Layer 2
        conv_up2 = Conv2DTranspose(filters=16 * 2**filter, kernel_size=2,
                                   strides=2,
                                   kernel_initializer=initializer)(
            up_block_1_6)
        conv_up_concat_2 = concatenate([conv_up2, down_block_3_4])

        up_block_2_1 = ResBlock(channels=16 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer,
                                decode=True)(conv_up_concat_2)
        up_block_2_2 = ResBlock(channels=16 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_2_1)
        up_block_2_3 = ResBlock(channels=16 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_2_2)
        up_block_2_4 = ResBlock(channels=16 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_2_3)

        # Layer 3
        conv_up3 = Conv2DTranspose(filters=8 * 2**filter, kernel_size=2, strides=2,
                                   kernel_initializer=initializer)(
            up_block_2_4)
        conv_up_concat_3 = concatenate([conv_up3, down_block_2_3])

        up_block_3_1 = ResBlock(channels=8 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer,
                                decode=True)(conv_up_concat_3)
        up_block_3_2 = ResBlock(channels=8 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_3_1)
        up_block_3_3 = ResBlock(channels=8 * 2**filter,
                                kernel_size=kernel_size,
                                activation=activation,
                                initializer=initializer)(up_block_3_2)

        # Layer 4
        conv_up4 = Conv2DTranspose(filters=8 * 2**filter, kernel_size=2,
                                   strides=2,
                                   kernel_initializer=initializer)(
            up_block_3_3)

        conv_up_concat_4 = concatenate([conv_up4, down1_activated])

        # need same padding so that output image size is the same as mask
        up_conv4 = Conv2D(filters=8 * 2**filter, kernel_size=7, strides=1,
                          kernel_initializer=initializer,
                          padding="same")(conv_up_concat_4)
        up4_activated = activation(up_conv4)
        up4_bn = BatchNormalization()(up4_activated)

        # Think about whether this needs to be activated
        conv_up5 = Conv2DTranspose(filters=8 * 2**filter, kernel_size=2, strides=2,
                                   kernel_initializer=initializer)(
            up4_bn)
        output_layer = Conv2D(self.output_channels, 1, strides=1,
                              padding='same',
                              activation="sigmoid")(conv_up5)

        model = tf.keras.Model(inputs=input, outputs=output_layer,
                               name="Hyper-UNetResnet")

        model.compile(
            optimizer=opt, loss=loss,
            metrics=['accuracy', tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall"),
                     tf.keras.metrics.AUC(name="auc", curve="PR")]
        )

        return model
