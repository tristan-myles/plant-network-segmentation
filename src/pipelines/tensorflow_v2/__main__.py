import datetime
import json
import logging

import numpy as np

from src.pipelines.tensorflow_v2.callbacks.lr_range_test import LRRangeTest
from src.pipelines.tensorflow_v2.callbacks.one_cycle import OneCycleLR
from src.pipelines.tensorflow_v2.helpers.train import get_tf_dataset, train
from src.pipelines.tensorflow_v2.helpers.utilities import (parse_arguments,
                                                           interactive_prompt,
                                                           format_input,
                                                           print_user_input)
from src.pipelines.tensorflow_v2.losses.custom_losses import *
from src.pipelines.tensorflow_v2.models.unet import Unet
from src.pipelines.tensorflow_v2.models.unet_resnet import UnetResnet
from src.pipelines.tensorflow_v2.models.wnet import WNet

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)


def main():
    ARGS = parse_arguments()

    if ARGS.interactive:
        ANSWERS = interactive_prompt()
    elif ARGS.json_path:
        with open(ARGS.json_path, "r") as JSON_FILE:
            ANSWERS = json.load(JSON_FILE)

    ANSWERS = format_input(ANSWERS)

    # convert to np.array for multi-element indexing
    METRICS = np.array([
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ])

    if ANSWERS["metric_choices"][0] == len(METRICS):
        metrics = METRICS
    else:
        metrics = METRICS[ANSWERS["metric_choices"]]

    callback_base_dir = "data/run_data/"

    lr_range_test = LRRangeTest(init_lr=1e-8, max_lr=3, iterations=12197,
                                verbose=1000)

    ocp = OneCycleLR(init_lr=1e-5, max_lr=1e-2, final_tail_lr=1e-9,
                     tail_length=0.2, iterations=243940)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='precision',
                                                      patience=8,
                                                      min_delta=0.001)
    csv_logger = tf.keras.callbacks.CSVLogger(
        f"{callback_base_dir}csv_logs/{ANSWERS['run_name']}.log",
        separator=',',  append=False)

    model_save_path = f"{callback_base_dir}saved_models/{ANSWERS['run_name']}"
    model_cpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=False, monitor='val_recall', mode='max',
        save_best_only=True)

    tb = tf.keras.callbacks.TensorBoard(
        log_dir=f"{callback_base_dir}tb_logs/{ANSWERS['run_name']}",
        histogram_freq=1, update_freq="epoch", profile_batch=0)

    CALLBACKS = np.array([
        lr_range_test, ocp, early_stopping, csv_logger, model_cpt, tb])

    if ANSWERS["callback_choices"][0] == len(CALLBACKS):
        callbacks = CALLBACKS
    else:
        callbacks = CALLBACKS[ANSWERS["callback_choices"]]

    # Input validations is done when answers are provided, hence the final
    # else statement as opposed to an elif
    if ANSWERS["loss_choice"] == 0:
        loss = tf.keras.losses.binary_crossentropy
    elif ANSWERS["loss_choice"] == 1:
        loss = weighted_CE(0.5)
    elif ANSWERS["loss_choice"] == 2:
        loss = focal_loss(0.5)
    else:
        loss = soft_dice_loss

    if ANSWERS["opt_choice"] == 0:
        opt = tf.keras.optimizers.Adam
    else:
        opt = tf.keras.optimizers.SGD

    if ANSWERS["model_choice"] == 0:
        model = Unet(1)
    elif ANSWERS["model_choice"] == 1:
        model = UnetResnet(1)
    else:
        model = WNet()

    # get dataset
    train_dataset, val_dataset = get_tf_dataset(
        train_base_dir=ANSWERS["train_base_dir"],
        val_base_dir=ANSWERS["val_base_dir"],
        leaf_ext=ANSWERS['leaf_ext'], mask_ext=ANSWERS['mask_ext'],
        incl_aug=ANSWERS['incl_aug'], batch_size=ANSWERS['batch_size'],
        buffer_size=ANSWERS['buffer_size'], leaf_shape=ANSWERS['mask_shape'],
        mask_shape=ANSWERS['mask_shape'])

    # train model
    _, model = train(train_dataset, val_dataset, list(metrics),
                    list(callbacks), model, ANSWERS["lr"], opt, loss,
                    ANSWERS["epochs"])

    model.save(model_save_path, save_format="tf")


if __name__ == "__main__":
    main()
