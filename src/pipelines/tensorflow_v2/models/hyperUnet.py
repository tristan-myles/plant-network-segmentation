from kerastuner import HyperModel
from kerastuner.tuners import Hyperband

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D,
                                     Conv2DTranspose, concatenate)


# *========================== U-Net Building Block ===========================*
class HyperUnet(HyperModel):
    def __init__(self, input_shape, output_channels):
        self.input_shape = input_shape
        self.output_channels = output_channels

    def build(self):
        input = Input(shape=self.input_shape)
        he_initializer = tf.keras.initializers.he_normal(seed=3141)
        activation = "relu"
        padding = "same"

        # Down 1
        conv1 = Conv2D(filters=64, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(input)
        conv1 = Conv2D(filters=64, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(conv1)
        # Half the x,y dimensions
        pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv1)
        conv2 = Conv2D(filters=128, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(pool1)
        conv2 = Conv2D(filters=128, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(conv2)

        pool2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2)
        conv3 = Conv2D(filters=256, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(pool2)
        conv3 = Conv2D(filters=256, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(conv3)

        pool3 = MaxPool2D(pool_size=(2, 2), strides=2)(conv3)
        conv4 = Conv2D(filters=512, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(pool3)
        conv4 = Conv2D(filters=512, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(conv4)

        #bottleneck
        pool4 = MaxPool2D(pool_size=(2, 2), strides=2)(conv4)
        conv5 = Conv2D(filters=1024, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(pool4)
        conv5 = Conv2D(filters=1024, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(conv5)

        #Upblock
        uptrans1 = Conv2DTranspose(filters=512,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=he_initializer)(conv5)
        skip1 = concatenate([conv4, uptrans1])
        up1 = Conv2D(filters=512, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(skip1)
        up1 = Conv2D(filters=512, kernel_size=3,
                     padding=padding, activation=activation,
                     kernel_initializer=he_initializer)(up1)

        uptrans2 = Conv2DTranspose(filters=256,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=he_initializer)(up1)
        skip2 = concatenate([conv3, uptrans2])
        up2 = Conv2D(filters=256, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(skip2)
        up2 = Conv2D(filters=256, kernel_size=3,
                     padding=padding, activation=activation,
                     kernel_initializer=he_initializer)(up2)

        uptrans3 = Conv2DTranspose(filters=128,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=he_initializer)(up2)
        skip3 = concatenate([conv2, uptrans3])
        up3 = Conv2D(filters=128, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(skip3)
        up3 = Conv2D(filters=128, kernel_size=3,
                     padding=padding, activation=activation,
                     kernel_initializer=he_initializer)(up3)

        uptrans4 = Conv2DTranspose(filters=64,
                                   kernel_size=2, strides=2,
                                   padding=padding,
                                   kernel_initializer=he_initializer)(up3)
        skip4 = concatenate([conv1, uptrans4])
        up4 = Conv2D(filters=64, kernel_size=3,
                       padding=padding, activation=activation,
                       kernel_initializer=he_initializer)(skip4)
        up4 = Conv2D(filters=64, kernel_size=3,
                     padding=padding, activation=activation,
                     kernel_initializer=he_initializer)(up4)

        output_layer = Conv2D(self.output_channels, 1, strides=1,
                              padding=padding,
                              activation="sigmoid",
                              name="classification_layer")(up4)

        model = tf.keras.Model(inputs=input, outputs=output_layer,
                               name="Hyper-UNet")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model


def tune_model(train_dataset, val_dataset, results_dir, run_name,
               input_shape, output_channels=1):
    hypermodel = HyperUnet(input_shape, output_channels)

    tuner = Hyperband(
        hypermodel,
        objective='val_recall',
        max_trials=1,
        directory=results_dir,
        project_name=run_name)

    tuner.search(x=train_dataset,
                 epochs=1,
                 validation_data=val_dataset)

    tuner.results_summary()
