.. _how_to_overview:

Overview
========
The plant handler component can be called by running src/__main__ as a module. There are
two modes of interaction: command line mode and interactive mode. These are
both explained in more detail in their respective how to use pages:
:ref:`how_to_cl` and :ref:`how_to_interact`. The
following actions are possible using the plant handler package:

#. Extract images
#. Extract tiles
#. Plot embolism profile
#. Plot embolism count barplot
#. EDA DataFrame
#. DataBunch DataFrame
#. Prediction (TF2 Model)
#. Trim sequence
#. Create Dataset
#. Augment Dataset

The tensorflow model component can be called by running
src/pipelines/tensorflow_v2/__main__ as a module. This can either be run in
interactive mode or by supplying a json file with the necessary instructions.
There is no command line mode in this case as there are many input
parameters. This is expanded on in this how to page: :ref:`how_to_tf2`.
