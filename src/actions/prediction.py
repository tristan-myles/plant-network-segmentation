import json
import logging
from typing import Dict, List, Tuple

from tensorflow import keras

from src.data_model.data_model import LeafSequence, MaskSequence
from src.helpers.utilities import classification_report
from src.pipelines.tensorflow_v2.losses.custom_losses import *
from src.pipelines.tensorflow_v2.models.unet import Unet
from src.pipelines.tensorflow_v2.models.unet_resnet import UnetResnet
from src.pipelines.tensorflow_v2.models.wnet import WNet

LOGGER = logging.getLogger(__name__)


# *================================ get model ================================*
def get_workaround_details(compilation_dict: Dict):
    """
    Creates a model using the saved compliation dict from training the TF
    model.

    :param compilation_dict: the compilation dict with the details to
     recreate the tensorflow model

    :return: a TF model, a TF loss, a TF optimiser, and TF metrics

    .. note::  This workaround is used since normal model saving for TF
     subclassed models did not work at the time of writing.

    """
    # model:
    if compilation_dict["model"] == "unet":
        model = Unet(1, compilation_dict["activation"],
                     compilation_dict["initializer"],
                     compilation_dict["filters"])
    elif compilation_dict["model"] == "unet_resnet":
        model = UnetResnet(1, compilation_dict["activation"],
                           compilation_dict["initializer"],
                           compilation_dict["filters"])
    elif compilation_dict["model"] == "wnet":
        model = WNet(1, compilation_dict["activation"],
                     compilation_dict["initializer"],
                     compilation_dict["filters"])
    else:
        raise ValueError("Please provide a valid answer for model choice, "
                         "options are unet, unet_resnet, or wnet")

    if compilation_dict["loss"] == "bce":
        loss = keras.losses.binary_crossentropy
    elif compilation_dict["loss"] == "wce":
        loss = WeightedCE(compilation_dict["loss_weight"])
    elif compilation_dict["loss"] == "focal":
        loss = FocalLoss(compilation_dict["loss_weight"])
    elif compilation_dict["loss"] == "dice":
        loss = SoftDiceLoss()
    else:
        raise ValueError("Please provide a valid answer for loss choice, "
                         "options are bce, wce, focal, or dice")

    if compilation_dict["opt"] == "adam":
        opt = keras.optimizers.Adam
    elif compilation_dict["opt"] == "sgd":
        opt = keras.optimizers.SGD
    else:
        raise ValueError("Please provide a valid answer for optimiser choice, "
                         "options are adam or sgd")

    metrics = [keras.metrics.BinaryAccuracy(name='accuracy'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall')]

    return model, loss, opt, metrics


# *=============================== prediction ================================*
def predict_tensorflow(lseq_list: List[LeafSequence],
                       model_weight_path: str,
                       leaf_shape: Tuple[int, int],
                       cr_csv_list: List[str] = "",
                       mseq_list: List[MaskSequence] = None,
                       threshold: float = 0.5,
                       format_dict: Dict = None) -> None:
    """
    Makes predictions for the images in each LeafSequence in the list of
    leaf sequences. The predictions are saved using a default path. If
    classification report csv save paths are provided, classification
    reports are generated and saved.

    :param lseq_list: a list of LeafSequence to make predictions for
    :param model_weight_path: the path to saved tf model to use to make
     predictions
    :param leaf_shape: the shape of leaf used when training the saved model
    :param cr_csv_list: file paths of where the classification reports
     should be saved; if this is provided, it must be the same length as the
     lseq_list
    :param mseq_list: the list of mseqs to use when generating the
     classification report, this must be provided if cr_csv_list is provided,
     and it must be the same length as lseq_list
    :param threshold: the threshold to use when saving predictions; i.e. a
      pixel is saved as an embolism if p(embolism) > threshold
    :param format_dict: the format to use when loading the LeafSequence Leaf
     images; these images are used as inputs for the predictions
    :return: None
    """

    if format_dict is None:
        format_dict = {}

    with open(model_weight_path + "compilation.json", "r") as json_file:
        compilation_dict = json.load(json_file)

    model, loss, opt, metrics = get_workaround_details(compilation_dict)
    model.load_workaround(leaf_shape, leaf_shape, loss,
                          opt(float(compilation_dict["lr"])), metrics,
                          model_weight_path)

    memory_saving = True
    cr_csv_list = cr_csv_list.split(";")

    if cr_csv_list[0]:
        memory_saving = False

    for i, lseq in enumerate(lseq_list):
        lseq.predict_leaf_sequence(model, leaf_shape[0],
                                   leaf_shape[1],
                                   memory_saving=memory_saving,
                                   leaf_shape=leaf_shape,
                                   threshold=threshold,
                                   **format_dict)

        if cr_csv_list[0]:
            mseq_list[i].load_extracted_images(load_image=True)

            temp_pred_list = []
            temp_mask_list = []

            for leaf, mask in zip(lseq.image_objects,
                                  mseq_list[i].image_objects):
                temp_pred_list.append(leaf.prediction_array / 255.0)
                temp_mask_list.append(mask.image_array / 255.0)

                # save memory
                del leaf.image_array
                del mask.image_array

            _ = classification_report(temp_pred_list, temp_mask_list,
                                      save_path=cr_csv_list[i])
