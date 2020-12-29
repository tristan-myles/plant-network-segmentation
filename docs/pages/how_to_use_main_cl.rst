.. _how_to_cl:

Package in Command Line Mode
============================
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
        },
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
The json template is the same as for the Embolism profile section.

EDA
---
EDA DF
^^^^^^
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
