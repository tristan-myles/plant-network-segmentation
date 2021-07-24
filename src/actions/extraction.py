import logging

LOGGER = logging.getLogger(__name__)


# *================================= images ==================================*
def extract_leaf_images(lseq_list, output_path_list, overwrite, format_dict):
    LOGGER.info("Extracting differenced leaf images")
    for lseq, output_path in zip(lseq_list, output_path_list):
        LOGGER.info(f"Differencing images in {lseq.folder_path} and saving "
                    f"to {output_path}")
        lseq.extract_changed_leaves(output_path, overwrite=overwrite,
                                    **format_dict)


def extract_multipage_mask_images(mseq_list, output_path_list,
                                  overwrite, binarise):
    LOGGER.info("Extracting mask images from multipage file")

    for mseq, output_path in zip(mseq_list, output_path_list):
        LOGGER.info(f"Extracting images from: {mseq.mpf_path} and saving "
                    f"to {output_path}")
        mseq.extract_mask_from_multipage(output_path,
                                         overwrite,
                                         binarise)

        # frees up ram when extracting many sequences
        mseq.unload_extracted_images()


def extract_tiles(seq_objects, length_x, stride_x, length_y,
                  stride_y, output_path_list=None, overwrite=False,
                  **kwargs):
    LOGGER.info(f"Extracting tiles from {seq_objects[0].__class__.__name__} "
                f"with the following configuration:"
                f" length (x, y): ({length_x}, {length_y}) |"
                f" stride (x, y): ({stride_x}, {stride_y})")

    if output_path_list is None:
        output_path_list = [None for _ in range(len(seq_objects))]

    for i, seq in enumerate(seq_objects):
        seq.load_image_array(**kwargs)

        seq.tile_sequence(length_x=length_x, stride_x=stride_x,
                          length_y=length_y, stride_y=stride_y,
                          output_path=output_path_list[i],
                          overwrite=overwrite)

        seq.unload_extracted_images()


# *=============================== DataFrames ================================*
def extract_full_eda_df(mseqs, options, output_path_list, lseqs=None):
    for i, (mseq, csv_output_path) in enumerate(zip(mseqs, output_path_list)):
        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images(load_image=True)

        if options["linked_filename"]:
            mseq.link_sequences(lseqs[i])

        _ = mseq.get_eda_dataframe(options, csv_output_path)

        mseq.unload_extracted_images()


def extract_tiles_eda_df(mseqs, options, output_path_list, lseqs=None):
    for i, (mseq, csv_output_path) in enumerate(zip(mseqs, output_path_list)):
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")

        if options["linked_filename"]:
            mseq.link_sequences(lseqs[i])

        _ = mseq.get_tile_eda_df(options, csv_output_path)

        mseq.unload_extracted_images()


def extract_full_databunch_df(lseqs, mseqs, output_path_list,
                              embolism_only=False):
    for csv_output_path, lseq, mseq in zip(output_path_list, lseqs, mseqs):
        mseq.load_extracted_images(load_image=True)
        mseq.get_embolism_percent_list()
        mseq.get_has_embolism_list()

        lseq.link_sequences(mseq)

        LOGGER.info(f"Creating DataBunch DataFrame using "
                    f"{lseq.__class__.__name__} located at {lseq.folder_path} "
                    f"and {mseq.__class__.__name__} located at"
                    f" {mseq.folder_path}")

        # get_databunch_dataframe written into of a lseq, i.e. will always
        # report lseq regardless of whether it is longer or shorter
        _ = lseq.get_databunch_dataframe(embolism_only, csv_output_path)


def extract_tiles_databunch_df(lseqs, mseqs, output_path_list,
                               tile_embolism_only=False,
                               leaf_embolism_only=False):
    for csv_output_path, lseq, mseq in zip(output_path_list, lseqs, mseqs):
        lseq.link_sequences(mseq)

        LOGGER.info(f"Creating Tile DataBunch DataFrame using "
                    f"{lseq.__class__.__name__} located at {lseq.folder_path} "
                    f"and {mseq.__class__.__name__} located at"
                    f" {mseq.folder_path}")

        _ = lseq.get_tile_databunch_df(mseq, tile_embolism_only,
                                       leaf_embolism_only,
                                       csv_output_path)
# *===========================================================================*
