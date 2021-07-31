from typing  import List, Tuple
import tensorflow as tf
import numpy as np
import logging
from src.model.model import Model
from src.pipelines.tensorflow_v2.helpers.utilities import (
    get_sorted_list, configure_for_performance, parse_image_fc,
    parse_numpy_image)

LOGGER = logging.getLogger(__name__)


# *============================ create tf dataset ============================*
def get_filepath_list(base_dir: str,
                      leaf_ext: str,
                      mask_ext: str,
                      incl_aug: bool = True) -> Tuple[List[str], List[str]]:
    """
    Gets a list of leaf and mask file paths. The folder structure needs to
    have been created using this code base.

    :param base_dir: base directory file path
    :param leaf_ext: the leaf extension
    :param mask_ext: the mask extension
    :param incl_aug: whether augmented images should be used
    :return: a list of leaf and mask file paths, as two separate lists
    """
    im_type = "embolism"
    leaf_dir = f"{base_dir}{im_type}/leaves/*.{leaf_ext}"
    mask_dir = f"{base_dir}{im_type}/masks/*.{mask_ext}"

    train_leaves = get_sorted_list(leaf_dir)
    train_masks = get_sorted_list(mask_dir)

    if incl_aug:
        im_type = "augmented"
        leaf_dir = f"{base_dir}{im_type}/leaves/*.{leaf_ext}"
        mask_dir = f"{base_dir}{im_type}/masks/*.{mask_ext}"

        aug_leaves = get_sorted_list(leaf_dir)
        aug_masks = get_sorted_list(mask_dir)
    else:
        aug_leaves = []
        aug_masks = []

    im_type = "no-embolism"
    leaf_dir = f"{base_dir}{im_type}/leaves/*.{leaf_ext}"
    mask_dir = f"{base_dir}{im_type}/masks/*.{mask_ext}"

    ne_leaves = get_sorted_list(leaf_dir)
    ne_masks = get_sorted_list(mask_dir)

    list_x = train_leaves + ne_leaves + aug_leaves
    list_y = train_masks + ne_masks + aug_masks

    assert len(list_x) == len(list_y), "X len != Y len"

    return list_x, list_y


def get_tf_dataset(base_dir: str,
                   leaf_ext: str,
                   mask_ext: str,
                   incl_aug:bool = False,
                   cfp:bool = True,
                   batch_size: int = 2,
                   buffer_size: int = 200,
                   leaf_shape: Tuple[int, int, int] = (512, 512, 1),
                   mask_shape: Tuple[int, int, int] = (512, 512, 1),
                   test: bool = False,
                   shift_256: bool = False,
                   transform_uint8: bool = False) -> \
        Tuple[tf.data.Dataset, List[str]]:
    """
    Creates a tf dataset, which has samples consisited of leaf, mask pairs.
    The dataset is also configured for performance. Both the dataset and
    list of leaf names are returned.

    :param base_dir: the base directory; the subfolders in this directory
     should contain the leaf and mask images the dataset will use
    :param leaf_ext: the leaf extensions
    :param mask_ext: the mask extensions
    :param incl_aug: whether augmented images should be included
    :param cfp: whether to configure the datasest for performance
    :param batch_size: the batch size
    :param buffer_size: the buffer size; this controls the extent of
     shuffling in the dataset
    :param leaf_shape: the leaf shape
    :param mask_shape: the mask shape
    :param test: whether a test set dataset is being configured
    :param shift_256: whether the leaf images should be shifted by 256
    :param transform_uint8: whether the leaf images should be transformed to a
     uint8 format
    :return: a tf dataset and a list of leaf file paths
    """
    data_x, data_y = get_filepath_list(base_dir, leaf_ext, mask_ext, incl_aug)

    parse_img_func = parse_image_fc(leaf_shape, mask_shape, test,
                                    shift_256, transform_uint8)

    dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
    dataset = dataset.map(
        parse_img_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if cfp:
        dataset = configure_for_performance(dataset, batch_size, buffer_size)

    return dataset, data_x


# *================================ TF Mixin =================================*
class _TfPnsMixin(Model):
    def train(self,
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              metrics: List[tf.keras.metrics.Metric],
              callbacks: List[tf.keras.callbacks.Callback],
              lr: float,
              opt: tf.keras.optimizers.Optimizer,
              loss: tf.keras.losses.Loss,
              epochs: int) -> tf.keras.callbacks.History:
        """
        Trains a tf.keras.model. The models weights will be updated in place by
         this method.

        :param train_dataset: the training dataset
        :param val_dataset: the validation dataset
        :param metrics: the metrics
        :param callbacks: the callbacks
        :param lr: the learning rate
        :param opt: the learning rate
        :param loss: the loss
        :param epochs: how many epochs to train the model for
        :return: a callback history object, which contains the details of
         the fitting.
        """
        self.compile(loss=loss,
                     optimizer=opt(lr=lr),
                     metrics=metrics)

        model_history = self.fit(train_dataset, epochs=epochs,
                                 validation_data=val_dataset,
                                 callbacks=[callbacks])

        return model_history

    def load_workaround(self,
                        mask_shape: Tuple[int, int, int],
                        leaf_shape: Tuple[int, int, int],
                        loss: tf.keras.losses.Loss,
                        opt: tf.keras.optimizers.Optimizer,
                        metrics: tf.keras.metrics.Metric,
                        chkpt_path: str) -> None:
        """
        Loads the model weights located at the provided path to the
        tf.keras.Model. All parameters must be the same as they were when
        generating the model weights to be loaded. There is no return from
        this function, but the weights of the model calling this function
        are updated.

        :param mask_shape: the mask shape
        :param leaf_shape: the leaf shape
        :param loss: the loss function
        :param opt: the optimiser
        :param metrics: the metrics
        :param chkpt_path: the filepath to the weights
        :return: None

        .. note::  This workaround is used since normal model saving for TF
         subclassed models did not work at the time of writing.
        """
        # Loading model weights, with the necessary workaround due to issues
        # using model.save and model.load with subclassed models, specifically
        # with custom objects
        x_train_blank = np.zeros((1,) + tuple(leaf_shape))
        y_train_blank = np.zeros((1,) + tuple(mask_shape))

        self.compile(loss=loss, optimizer=opt, metrics=metrics)

        # This initializes the variables used by the optimizers,
        # as well as any stateful metric variables
        self.train_on_batch(x_train_blank, y_train_blank)

        # Load the state of the old model
        self.load_weights(chkpt_path)

    def predict_tile(self,
                     new_tile: np.array,
                     leaf_shape: Tuple[int, int, int],
                     post_process: bool = True) -> np.array:
        """
        Predicts a tile using a trained tf.keras.Model. This method is
        required by the Model class this Mixin inherits.

        :param new_tile: the tile to use as input
        :param leaf_shape: the leaf tile shape
        :param post_process: whether the prediction should be post processed
        :return: the prediction
        """
        batch_shape = (1,) + leaf_shape
        img = parse_numpy_image(new_tile, batch_shape)
        prediction = self.predict(img)
        prediction = np.reshape(prediction, leaf_shape)

        # post-process only if shifted by 256:
        if post_process:
            prediction[new_tile >= 0] = 0

        return prediction
# *===========================================================================*
