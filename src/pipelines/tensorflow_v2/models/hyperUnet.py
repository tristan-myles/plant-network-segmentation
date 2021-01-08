import tensorflow as tf
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

from tensorflow.keras.layers import Input, Conv2D

from src.pipelines.tensorflow_v2.models.unet import UnetBlock


class HyperUnet(HyperModel):
    def __init__(self, input_shape, output_channels):
        self.input_shape = input_shape
        self.output_channels = output_channels

    def build(self, hp):
        x = Input(shape=self.input_shape)

        he_initializer = tf.keras.initializers.he_normal(seed=3141)

        # Contracting
        # Layer 1
        conv_down1 = UnetBlock(num_filters=64, kernel_size=3, decode=True,
                               initializer=he_initializer,
                               name="down1")(x)

        # Layer 2
        conv_down2 = UnetBlock(num_filters=128, kernel_size=3,
                               decode=True,
                               initializer=he_initializer,
                               name="down2")(conv_down1)

        # Layer 3
        conv_down3 = UnetBlock(num_filters=256, kernel_size=3,
                               decode=True,
                               initializer=he_initializer,
                               name="down3")(conv_down2)

        # Layer 4
        conv_down4 = UnetBlock(num_filters=512, kernel_size=3,
                               decode=True,
                               initializer=he_initializer,
                               name="down4")(conv_down3)

        # Bottleneck
        conv_bottle = UnetBlock(num_filters=1024, kernel_size=3,
                                     initializer=he_initializer,
                                     name="bottleneck")(conv_down4)

        # Expanding

        # Layer 1
        # No activation ... Since skip happens before the activation
        conv_up1 = UnetBlock(num_filters=512, kernel_size=3, encode=True,
                             initializer=he_initializer,
                             name="up1")(conv_bottle, conv_down4)
        # Layer 2
        conv_up2 = UnetBlock(num_filters=256, kernel_size=3, encode=True,
                             initializer=he_initializer,
                             name="up2")(conv_up1, conv_down3)
        # Layer 3
        conv_up3 = UnetBlock(num_filters=128, kernel_size=3, encode=True,
                             initializer=he_initializer,
                             name="up3")(conv_up2, conv_down2)
        # Layer 4
        conv_up4 = UnetBlock(num_filters=64, kernel_size=3, encode=True,
                             initializer=he_initializer,
                             name="up4")(conv_up3, conv_down1)

        # Output
        output_layer = Conv2D(self.output_channels, 1, strides=1,
                              padding='same',
                              activation="sigmoid",
                              name="classification_layer")(conv_up4)

        model = tf.keras.Model(x, output_layer, name="Hyper-UNet")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='cross-entropy',
            metrics=['accuracy']
        )

        return model


def tune_model(train_dataset, val_dataset, results_dir, run_name,
               input_shape, output_channels=1):
    hypermodel = HyperUnet(input_shape, output_channels)

    tuner = RandomSearch(
        hypermodel,
        objective='val_recall',
        max_trials=1,
        directory=results_dir,
        project_name=run_name)

    tuner.search(x=train_dataset,
                 epochs=1,
                 validation_data=val_dataset)

    tuner.results_summary()




