import json
import logging
from pathlib import Path

import numpy as np
import os
import random

from src.helpers.utilities import classification_report
from src.pipelines.tensorflow_v2.callbacks.lr_range_test import LRRangeTest
from src.pipelines.tensorflow_v2.callbacks.one_cycle import OneCycleLR
from src.pipelines.tensorflow_v2.helpers.train_test import get_tf_dataset
from src.pipelines.tensorflow_v2.helpers.utilities import (
                                                        parse_arguments,
                                                        interactive_prompt,
                                                        im2_lt_im1,
                                                        save_compilation_dict,
                                                        tune_model,
                                                        save_lrt_results)
from src.pipelines.tensorflow_v2.losses.custom_losses import *
from src.pipelines.tensorflow_v2.models.unet import Unet
from src.pipelines.tensorflow_v2.models.hyper_unet import HyperUnet
from src.pipelines.tensorflow_v2.models.unet_resnet import UnetResnet
from src.pipelines.tensorflow_v2.models.hyper_unet_resnet import (
                                                            HyperUnetResnet)
from src.pipelines.tensorflow_v2.models.wnet import WNet
from src.pipelines.tensorflow_v2.models.hyper_wnet import HyperWnet

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)

seed = 3141
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def main():
    ARGS = parse_arguments()

    if ARGS.interactive:
        ANSWERS = interactive_prompt()
    elif ARGS.json_path:
        with open(ARGS.json_path, "r") as JSON_FILE:
            ANSWERS = json.load(JSON_FILE)

    OUTPUTS_DIR = "data/run_data/"

    # get dataset
    # train
    train_dataset = get_tf_dataset(
        base_dir=ANSWERS["train_base_dir"],
        leaf_ext=ANSWERS['leaf_ext'], mask_ext=ANSWERS['mask_ext'],
        incl_aug=ANSWERS['incl_aug'], batch_size=ANSWERS['batch_size'],
        buffer_size=ANSWERS['buffer_size'],
        leaf_shape=ANSWERS['leaf_shape'],
        mask_shape=ANSWERS['mask_shape'],
        train=True,
        shift_256=ANSWERS['shift_256'],
        transform_uint8=ANSWERS['transform_uint8'])

    # val
    val_dataset = get_tf_dataset(
        base_dir=ANSWERS["val_base_dir"],
        leaf_ext=ANSWERS['leaf_ext'], mask_ext=ANSWERS['mask_ext'],
        batch_size=ANSWERS['batch_size'],
        buffer_size=None,
        leaf_shape=ANSWERS['leaf_shape'], mask_shape=ANSWERS['mask_shape'],
        shift_256=ANSWERS['shift_256'],
        transform_uint8=ANSWERS['transform_uint8'])

    if ANSWERS["which"] == "tuning":
        if ANSWERS["model_choice"] == 0:
            hyper_model = HyperUnet
        elif ANSWERS["model_choice"] == 1:
            hyper_model = HyperUnetResnet
        else:
            hyper_model = HyperWnet

        tune_model(hyper_model, train_dataset, val_dataset,
                   OUTPUTS_DIR, ANSWERS["run_name"],
                   ANSWERS["leaf_shape"])

    elif ANSWERS["which"] == "training":
        if ANSWERS["model_choice"] == 0:
            model = Unet(1)
        elif ANSWERS["model_choice"] == 1:
            model = UnetResnet(1)
        else:
            model = WNet()

        model_save_path = (f"{OUTPUTS_DIR}saved_models/"
                           f"{ANSWERS['run_name']}")
        model_save_path = model_save_path

        # Input validations is done when answers are provided, hence the final
        # else statement as opposed to an elif
        if ANSWERS["loss_choice"] == 0:
            loss = tf.keras.losses.binary_crossentropy
        elif ANSWERS["loss_choice"] == 1:
            loss = weighted_CE(0.5)
        elif ANSWERS["loss_choice"] == 2:
            loss = focal_loss(0.5)
            model_save_path = model_save_path
        else:
            loss = soft_dice_loss
            model_save_path = model_save_path

        if ANSWERS["opt_choice"] == 0:
            opt = tf.keras.optimizers.Adam
        else:
            opt = tf.keras.optimizers.SGD

        if ANSWERS["callback_choices"]:

            lr_range_test = LRRangeTest(init_lr=1e-8, max_lr=3,
                                        iterations=12197, verbose=1000)

            ocp = OneCycleLR(init_lr=1e-5, max_lr=1e-2, final_tail_lr=1e-9,
                             tail_length=0.2, iterations=243940)

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='precision', patience=8, min_delta=0.001)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{OUTPUTS_DIR}csv_logs/{ANSWERS['run_name']}.log",
                separator=',', append=False)

            model_save_path = model_save_path + "/"
            Path(model_save_path).mkdir(parents=True, exist_ok=True)

            model_cpt = tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path, save_weights_only=True, mode='max',
                monitor='val_recall', save_best_only=True, save_format="tf")

            tb = tf.keras.callbacks.TensorBoard(
                log_dir=f"{OUTPUTS_DIR}tb_logs/{ANSWERS['run_name']}",
                histogram_freq=1, update_freq="epoch", profile_batch=0)

            CALLBACKS = np.array([
                lr_range_test, ocp, early_stopping, csv_logger, model_cpt, tb])

            if ANSWERS["callback_choices"][0] == len(CALLBACKS):
                callbacks = list(CALLBACKS)
            else:
                callbacks = list(CALLBACKS[ANSWERS["callback_choices"]])
        else:
            callbacks = []

        # convert to np.array for multi-element indexing
        METRICS = np.array([
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc', curve="PR"),
            tf.keras.metrics.MeanIoU(2, name="IoU")])

        if ANSWERS["metric_choices"][0] == len(METRICS):
            metrics = list(METRICS)
        else:
            metrics = list(METRICS[ANSWERS["metric_choices"]])

        # train model
        _ = model.train(train_dataset, val_dataset, metrics,
                        callbacks, ANSWERS["lr"], opt, loss,
                        ANSWERS["epochs"])

        # save the compilation dict once training is complete
        lr = model.optimizer.lr.numpy()
        save_compilation_dict(ANSWERS, lr, model_save_path)

        if 0 in ANSWERS['callback_choices']:
            lrt_save_path = f"{OUTPUTS_DIR}lrt/"f"{ANSWERS['run_name']}"
            Path(lrt_save_path ).mkdir(parents=True, exist_ok=True)
            save_lrt_results(lr_range_test, lrt_save_path)

        if 4 in ANSWERS["callback_choices"]:
            # load the best model and check that it has the same validation
            # recall as the best val recall achieved during training
            del model
            if ANSWERS["model_choice"] == 0:
                model = Unet(1)
            elif ANSWERS["model_choice"] == 1:
                model = UnetResnet(1)
            else:
                model = WNet()

            # load model with best val recall
            model.load_workaround(ANSWERS["mask_shape"], ANSWERS['leaf_shape'],
                                  loss, opt(lr), metrics, model_save_path)

        # check that recall is the same as the best val recall during training
        LOGGER.info("Confirming that best model saved correctly: ")
        model.evaluate(val_dataset)

        # check test_set
        if ANSWERS["test_dir"]:
            # test
            test_dataset = get_tf_dataset(
                base_dir=ANSWERS["test_dir"],
                leaf_ext=ANSWERS['leaf_ext'], mask_ext=ANSWERS['mask_ext'],
                batch_size=1,
                buffer_size=None,
                leaf_shape=ANSWERS['leaf_shape'],
                mask_shape=ANSWERS['mask_shape'],
                shift_256=ANSWERS['shift_256'],
                transform_uint8=ANSWERS['transform_uint8'])

            LOGGER.info("Test set")
            model.evaluate(test_dataset)

            # Could have a memory issue by keeping all the predictions and all
            # the masks in memory at the same time, possibly better to use a
            # generator in the classification report...
            predictions = model.predict(test_dataset)

            masks = []
            leaves = []
            for imageset in test_dataset.as_numpy_iterator():
                masks.append(imageset[1][0])
                leaves.append(imageset[0][0])

            masks = [imageset[1] for imageset in
                     test_dataset.as_numpy_iterator()]

            csv_save_path = (f"{OUTPUTS_DIR}classification_reports/"
                             f"{ANSWERS['run_name']}.csv")

            _ = classification_report(predictions, masks,
                                      save_path=csv_save_path)

            predictions = [im2_lt_im1(pred, leaf) for pred, leaf
                           in zip(predictions, leaves)]

            csv_save_path = (csv_save_path.rsplit(".", 1)[0] +
                             "_post_processed.csv")
            _ = classification_report(predictions, masks,
                                      save_path=csv_save_path)


if __name__ == "__main__":
    main()
