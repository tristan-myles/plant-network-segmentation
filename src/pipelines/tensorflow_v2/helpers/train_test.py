import tensorflow as tf

from src.data.data_model import *
from src.model.model import Model
from src.pipelines.tensorflow_v2.helpers.utilities import (
    get_sorted_list, configure_for_performance, parse_image_fc)

LOGGER = logging.getLogger(__name__)


# *============================ create tf dataset ============================*
def get_filepath_list(base_dir, leaf_ext, mask_ext, incl_aug=True):
    im_type = "embolism"
    leaf_dir = f"{base_dir}{im_type}/leaves/*.{leaf_ext}"
    mask_dir = f"{base_dir}{im_type}/masks/*.{mask_ext}"

    train_leaves = get_sorted_list(leaf_dir)
    train_masks = get_sorted_list(mask_dir)

    if incl_aug:
        im_type = "augmented"
        aug_leaves = get_sorted_list(leaf_dir)
        aug_masks = get_sorted_list(mask_dir)
    else:
        aug_leaves = []
        aug_masks = []

    im_type = "no-embolism"
    ne_leaves = get_sorted_list(leaf_dir)
    ne_masks = get_sorted_list(mask_dir)

    list_x = train_leaves + ne_leaves + aug_leaves
    list_y = train_masks + ne_masks + aug_masks

    assert len(list_x) == len(list_y), "X len != Y len"

    return list_x, list_y


def get_tf_dataset(train_base_dir, val_base_dir, leaf_ext, mask_ext,
                   incl_aug, cfp=True, batch_size=2, buffer_size=200,
                   leaf_shape=(512, 512, 1), mask_shape=(512, 512, 1)):
    train_x, train_y = get_filepath_list(train_base_dir, leaf_ext, mask_ext,
                                         incl_aug)

    val_x, val_y = get_filepath_list(val_base_dir, leaf_ext, mask_ext,
                                     incl_aug=False)

    parse_img_func = parse_image_fc(leaf_shape, mask_shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.map(
        parse_img_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_dataset = val_dataset.map(
        parse_img_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if cfp:
        train_dataset = configure_for_performance(train_dataset,
                                                  batch_size, buffer_size)
        val_dataset = configure_for_performance(val_dataset, batch_size,
                                                buffer_size)

    return train_dataset, val_dataset


# *================================ TF Mixin =================================*
class _TfPnsMixin(Model):
    def train(self, train_dataset, val_dataset, metrics, callbacks, lr,
              opt, loss, epochs):
        self.compile(loss=loss,
                     optimizer=opt(lr=lr),
                     metrics=metrics)

        model_history = self.fit(train_dataset, epochs=epochs,
                                 validation_data=val_dataset,
                                 callbacks=[callbacks])

        return model_history

    def predict_tile(self, new_tile):
        pass
# *===========================================================================*
