How to Use
==========

Extraction
----------

Python Command
^^^^^^^^^^^^^^

.. prompt:: bash $
   
   python -m src.__main__ <input.json> extract_images

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


Input Json Template
^^^^^^^^^^^^^^^^^^^

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
------

Plotting
--------

Prediction
----------
.. code-block:: json
