.. _how_to_cl:

Plant Handler in Command Line Mode
==================================
The command line mode allows the user to interact with the plant handler
component using the command line arguments. In addition to preference, this
may be helpful if the script needs to be run using a bash script, for example.
Specifying where to find leaf and mask sequences is likely to be verbose,
hence, this information is usually specified using a json find. This can be
provided using the the -j argument, not to be confused with the -fj flag. In
addition it is possible to provide other argument which may be too verbose in
this json file; output names are an example. When this is done, you will
usually input same for that argument via the relevant command line flag.

The input jsons, flag explanations, and additional notes, where necessary, for
each action are provided below.

Extraction
----------
Images
^^^^^^
Python Command
""""""""""""""

.. prompt:: bash $

   python -m src.__main__ -fj <input.json> extract_images

usage:
       Perform operations using the plant-image-segmentation code base json_path extract_images

optional arguments:
  -h, --help            show this help message and exit
  --leaf_output_path, -lo
                        output paths, if the paths are in the input json enter
                        "same"
  --mask_output_path, -mo
                        output paths, if the paths are in the input json enter
                        "same"
  --overwrite, -o       overwrite existing images, note this flag is applied
                        to both mask and leaf images
  --binarise, -b        save binary masks


JSON template
"""""""""""""
.. code-block:: json

   {
    "leaves": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
        }
       "format": {
         "shift_256": true
       }
        "output": {
        "output_path": []
        }
    },
    "masks": {
        "input": {
        "mpf_path": []
       },
        "output": {
        "output_path": []
        }
    }
   }

Tiling
^^^^^^
.. warning::
    Using custom file names to save the tile is not currently working.
    Rather use "default"

Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> extract_tiles

usage: Perform operations using the plant-image-segmentation code base json_path extract_tiles

optional arguments:
  -h, --help            show this help message and exit
  -sx, --stride_x   x stride size
  -sy, --stride_y   y stride size
  -lx, --length_x   tile x length
  -ly, --length_y   tile y length
  --leaf_output_path, -lo
                        output paths, if you want to use the default path
                        enter "default", if the paths are in the input json
                        enter "same"
  --mask_output_path, -mo
                        output paths, if you want to use the default path
                        enter "default", if the paths are in the input json
                        enter "same"

JSON template
"""""""""""""

.. code-block:: json

   {
    "leaves": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
        },
       "format": {
         "shift_256": true
       }
        "output": {
        "output_path": []
        }
    },
    "masks": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
       },
        "output": {
        "output_path": []
        }
    }
   }


Plotting
--------
Embolism profile
^^^^^^^^^^^^^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> plot_profile

usage: Perform operations using the plant-image-segmentation code base plot_profile

optional arguments:
  -h, --help            show this help message and exit
  --output_path, -o    The plot output path
  --show, -s    flag indicating if the plot should be shown
  --leaf_names, -ln    leaf names to be used in plot title


JSON template
"""""""""""""
.. code-block:: json

    {
      "masks": {
        "input": {
          "folder_path": [
            "/test1/test2/test3/extracted_images/masks/",
            "/test4/test5/extracted_images/masks/"
          ],
          "filename_pattern": [
            "*.png",
            "*.png"
          ]
        }
      },
      "plots": {
        "output_paths": [
          "test3.svg",
          "test2.svg"
        ],
        "leaf_names": [
          "Leaf 1",
          "Leaf 2"
        ]
      }
    }



Embolism count
^^^^^^^^^^^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> plot_embolism_counts

usage: Perform operations using the plant-image-segmentation code base plot_embolism_counts

optional arguments:
  -h, --help            show this help message and exit
  --output_path, -o    The plot output path
  --show, -s    flag indicating if the plot should be shown
  --leaf_names, -ln    leaf names to be used in plot title
  --tile, -t    indicates if the plot should be created using tiles
  --leaf_embolism_only, -leo    should only full leaves with embolisms be used
  --percent, -p    should the plot y-axis be expressed as a percent

The json template is the same as for the Embolism profile section.

EDA
---
EDA DF
^^^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> eda_df

usage: Perform operations using the plant-image-segmentation code base eda_df

positional arguments:
  csv_output_path    output paths, if the paths are in the input json enter "same"

optional arguments:
  -h, --help    show this help message and exit
  --tiles, -t    whether tiles should be used

JSON template
"""""""""""""
.. code-block:: json

    {
      "leaves": {
        "input": {
          "folder_path": [
            "/test1/test2/test3/extracted_images/leaves/",
            "/test4/test5/extracted_images/leaves/"
          ],
          "filename_pattern": [
            "*.png",
            "*.png"
          ]
        }
       "format": {
         "shift_256": true
       }
      },
      "masks": {
        "input": {
          "folder_path": [
            "/test1/test2/test3/extracted_images/masks/",
            "/test4/test5/extracted_images/masks/"
          ],
          "filename_pattern": [
            "*.png",
            "*.png"
          ]
        }
      },
      "eda_df": {
        "options": {
          "linked_filename": true,
          "unique_range": true,
          "embolism_percent": true,
          "intersection": true,
          "has_embolism": true
        },
        "output_path": [
          "test3_eda.csv",
          "test5_eda.csv"
        ]
      }
    }



DataBunch DF
^^^^^^^^^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> databunch_df

usage: Perform operations using the plant-image-segmentation code base databunch_df

positional arguments:
  csv_output_path       output paths, if the paths are in the input json enter "same"

optional arguments:
  -h, --help            show this help message and exit
  --tiles, -t    whether tiles should be used
  --tile_embolism_only, -teo    should only tiles with embolisms be used
  --leaf_embolism_only, -leo    should only full leaves with embolisms be used

JSON template
"""""""""""""
.. code-block:: json

    {
      "leaves": {
        "input": {
          "folder_path": [
            "/test1/test2/test3/extracted_images/leaves/",
            "/test4/test5/extracted_images/leaves/"
          ],
          "filename_pattern": [
            "*.png",
            "*.png"
          ]
        }
       "format": {
         "shift_256": true
       }
      },
      "masks": {
        "input": {
          "folder_path": [
            "/test1/test2/test3/extracted_images/masks/",
            "/test4/test5/extracted_images/masks/"
          ],
          "filename_pattern": [
            "*.png",
            "*.png"
          ]
        }
      },
      "databunch_df": {
        "output_path": [
          "test3_databunch.csv",
          "test5_databunch.csv"
        ]
      }
    }


General
-------
Trim
^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> trim_sequence

usage: Perform operations using the plant-image-segmentation code base trim_sequence

optional arguments:
  -h, --help            show this help message and exit
  --mask, -m    whether the mask sequence should be trimmed, default is for the leaf sequence to be trimmed
  --y_size_dir, -ysd    y output size and direction to be passed in as a tuple, where a 1 or -1 indicated to trim either top or bottom respectively
  --x_size_dir, -xsd    x output size and direction to be passed in as a tuple, where a 1 or -1 indicated to trim either left or right respectively
  --overwrite, -o    whether or not the image being trimmed should be overwritten


JSON template
"""""""""""""
.. code-block:: json

    {
      "leaves": {
        "input": {
          "folder_path": [
            "/test1/test2/test3/extracted_images/leaves/",
            "/test4/test5/extracted_images/leaves/"
          ],
          "filename_pattern": [
            "*.png",
            "*.png"
          ]
        }
       "format": {
         "shift_256": true
       }
      },
      "masks": {
        "input": {
          "folder_path": [
            "/test1/test2/test3/extracted_images/masks/",
            "/test4/test5/extracted_images/masks/"
          ],
          "filename_pattern": [
            "*.png",
            "*.png"
          ]
        }
      },
      "trim": {
        "x_size_dir": [
          [1000, -1],
          [1440, 1 ]
        ],
        "y_size_dir": [
          [1000, 1],
          null
        ]
      }
    }

Predict
-------
TF 2 Model
^^^^^^^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> predict

usage: Perform operations using the plant-image-segmentation code base predict

optional arguments:
  -h, --help            show this help message and exit
  --model_path, -mp   the path to the saved model weights to restore
  --threshold, -t   classification threshold to determine if a pixel is an embolism or not.
  --csv_path, -cp   csv path of where the classification report should be saved; this flag determines if a classification report is generated
  --leaf_shape, -ls   leaf shape, please separate each number by a ';'

JSON template
"""""""""""""
.. code-block:: json

   {
    "leaves": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
        },
       "format": {
         "shift_256": true
       }
    },
    "masks": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
       },
    }
   }

Dataset
-------
Create Dataset
^^^^^^^^^^^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> create_dataset

usage: Perform operations using the plant-image-segmentation code base create_dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path, -dp    the path where the dataset should be created,including the dataset name
  --downsample_split, -ds    the fraction of non-embolism images to remove
  --test_split, -ts    the fraction of the data to use for a test set
  --val_split, -vs    the fraction of the data to use for a val set
  --lolo, -l    the leaf to leave out for the test set


JSON template
"""""""""""""
.. code-block:: json

   {
    "leaves": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
        },
       "format": {
         "shift_256": true
       }
    },
    "masks": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
       },
    }
   }


Augment Dataset
^^^^^^^^^^^^^^^
Python Command
""""""""""""""
.. prompt:: bash $

  python -m src.__main__ -fj <input.json> augment_dataset

There are no optional arguments

JSON template
"""""""""""""
.. code-block:: json

   {
    "leaves": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
        },
       "format": {
         "shift_256": true
       }
    },
    "masks": {
        "input": {
        "folder_path": [],
        "filename_pattern": []
       },
    }
   }
