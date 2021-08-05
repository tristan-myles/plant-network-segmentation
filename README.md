# Plant Network Segmentation

Welcome to the home of the Plant Network Segmentation code base. This code 
base can be used to interact with raw leaf images intended to be used as 
input to the Optical Vulnerability (OV) method. 

<p align="center">
  <img src="docs/resources/images/leaf.gif?raw=True" width="380" height="325" >
  <img src="docs/resources/images/vc.gif?raw=True" wwidth="325" height="325" >
</p>

The code base is broken into two components: a plant handler component and a 
TF2 model builder component.

The plant handler contains actions to interact with OV method data. In 
addition, it contains an object orientated data model to make further analysis 
easier. To interact with the plant handler use:

```python
python -m src.__main__ -i
```

The TF2 model builder can be used to train a Tensorflow model. Three models are 
available for training, namely U-Net, U-Net (ResNet34), and W-Net.
The trained model can be  saved and then used to make 
predictions in the first component. However, the first component has been 
structured such that the prediction is independent of Tensorflow specifically. 
Any model which inherits the abstract Model class, and consequently implements a 
predict_tile method can be used. The how to use section explains how to interact
with each component. To interact with the model builder use:

```python
python -m src.pipelines.tensorflow_v2.__main__ -i
```

For more detailed documentation please visit: https://plant-network-segmentation.readthedocs.io/en/latest/