import logging
import logging.config
from little_helpers import extract_dataset
from PIL import ImageChops
from os import path
from glob import glob
abs_path = path.dirname(path.abspath(__file__))

logging.config.fileConfig(fname=abs_path + "/logging_configuration.ini",
                          defaults={'logfilename': abs_path + "/main.log"},
                          disable_existing_loggers=False)
LOGGER = logging.getLogger(__name__)

def extract_masks_and_leaves():
    common_path = "/mnt/disk3/thesis/data"

    folder_path_list = ["0_qk1_1", "1_qk3", "2_qtom1", "3_qgam1", "4_qp1",
                        "5_qp2", "6_qp4", "7_qp5"]

    mask_seq_name_list = ["Mask of Result of Substack (2-344).tif",
                          "Mask of Result of Substack (2-743).tif",
                          "Result of Substack (2-567).tif",
                          "Mask of Result of Substack (2-644).tif",
                          "Mask of Result of Substack (2-397).tif",
                          "Mask of Result of Substack (2-208).tif",
                          "Mask of Result of Substack (2-738).tif",
                          "Result of Substack (2-704).tif"]

    for i, (leaf_path, mask_seq_name) in enumerate(zip(folder_path_list,
                                                       mask_seq_name_list)):
        folder_path = f"{common_path}/{leaf_path}/"

        # Leaf extraction
        file_names = sorted([f for f in glob(folder_path + "2019*.tif",
                                             recursive=True)])
        diff_leaves_output_path = f"{folder_path}tristan/diffs/"
        diff_name = f"leaf_{i}_diff.tif"
        diff_path_list = extract_dataset.extract_changed_sequence(
            file_names, diff_leaves_output_path, diff_name, overwrite=True)
        LOGGER.info(f"{len(diff_path_list)} difference images created")

        # Leaf extraction 2
        file_names = sorted([f for f in glob(folder_path + "2019*.tif",
                                             recursive=True)])
        diff_leaves_output_path = f"{folder_path}tristan/diffs_from_init/"
        diff_name = f"leaf_{i}_diff.tif"
        diff_path_list = extract_dataset.extract_changed_sequence(
            file_names, diff_leaves_output_path, diff_name, dif_len=0,
            overwrite=True)
        LOGGER.info(f"{len(diff_path_list)} difference images created")

        # Mask extraction
        mask_output_folder_path = f"{folder_path}tristan/masks/"

        mask_path_list = extract_dataset.extract_images_from_image_sequence(
            f"{folder_path}{mask_seq_name}", mask_output_folder_path,
            f"leaf_{i}_mask.png")
        LOGGER.info(f"{len(mask_path_list)} mask images created")

        # Binary extraction
        mask_output_folder_path = f"{folder_path}tristan/binary_masks/"

        mask_path_list = extract_dataset.extract_images_from_image_sequence(
            f"{folder_path}{mask_seq_name}", mask_output_folder_path,
            f"leaf_{i}_binary_mask.png", binarise=True)
        LOGGER.info(f"{len(mask_path_list)} binary mask images created")

        # add masks
        file_names = sorted([f for f in glob(f"{folder_path}tristan/masks/"
                                             + "*.png", recursive=True)])
        diff_leaves_output_path = f"{folder_path}tristan/combined_masks/"
        diff_name = f"leaf_{i}_mask_overlay.png"
        diff_path_list = extract_dataset.extract_changed_sequence(
            file_names, diff_leaves_output_path, diff_name, dif_len=0,
            combination_function=ImageChops.add_modulo,
            sequential=True,
            overwrite=True)
        LOGGER.info(f"{len(diff_path_list)} overlaid mask images created")


if __name__ == "__main__":
    extract_masks_and_leaves()
