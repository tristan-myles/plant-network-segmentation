import tensorflow as tf

from src.data.data_model import *
from src.model.model import Model
from src.pipelines.tensorflow_v2.helpers.utilities import (
    get_sorted_list, configure_for_performance, parse_image_fc,
    parse_numpy_image)

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


def get_tf_dataset(base_dir, leaf_ext, mask_ext,
                   incl_aug=False, cfp=True, batch_size=2, buffer_size=200,
                   leaf_shape=(512, 512, 1), mask_shape=(512, 512, 1),
                   train=False):

    data_x, data_y = get_filepath_list(base_dir, leaf_ext, mask_ext, incl_aug)

    parse_img_func = parse_image_fc(leaf_shape, mask_shape, train)

    dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
    dataset = dataset.map(
        parse_img_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if cfp:
        dataset = configure_for_performance(dataset, batch_size, buffer_size)

    return dataset


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

    def load_workaround(self, mask_shape, leaf_shape, loss, opt, metrics,
                        chkpt_path):
        # Loading model weights, with the necessary workaround due to issues
        # using model.save and model.load with subclassed models, specifically
        # with custom objects
        x_train_blank = np.zeros((1,) + leaf_shape)
        y_train_blank = np.zeros((1,) + mask_shape)

        self.compile(loss=loss, optimizer=opt, metrics=metrics)

        # This initializes the variables used by the optimizers,
        # as well as any stateful metric variables
        self.train_on_batch(x_train_blank, y_train_blank)

        # Load the state of the old model
        self.load_weights(chkpt_path)

    def predict_tile(self, new_tile, leaf_shape, post_process=True):
        batch_shape = (1,) + leaf_shape
        img = parse_numpy_image(new_tile, batch_shape)
        prediction = self.predict(img)
        prediction = np.reshape(prediction, leaf_shape)

        # post-process only if shifted by 256:
        if post_process:
            prediction[new_tile >= 0] = 0
        prediction = prediction * 255

        return prediction
# *===========================================================================*
