# I3D models trained on Kinetics

## Overview

This repository contains trained models reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman.

This code is based on Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d). Including PyTorch versions of their models.

## Note
This code was written for PyTorch 0.3. Version 0.4 and newer may cause issues.
More detailed information about the charaders dataset could refer the link: https://allenai.org/plato/charades/README.txt 

## Video Proprocessing
For different mode we have differnt methods to preprocess the videos. For mode is "rgb", we need to transform the videos to images by using the tools [ffmpeg](ffmpeg), and we put the images in the directory [data/changing_tire_00002_rgb](/data/changing_tire_00002_rgb) and [/data/changing_tire_00003_rgb](/data/changing_tire_00003_rgb).

For mode is "flow", we use need to use dense-flow.py to generate images which may take 30 minites to finish.  We set the maximum of optical flows to 20 referred by the https://allenai.org/plato/charades/README.txt. We put all the images in the directory [/data/flows/changing_tire_0002](/data/flows/changing_tire_0002) and [/data/flows/changing_tire_0003](/data/flows/changing_tire_0003).


# Fine-tuning and Feature Extraction
We provide code to extract I3D features and fine-tune I3D for charades. Our fine-tuned models on charades are also available in the models director (in addition to Deepmind's trained models). The deepmind pre-trained models were converted to PyTorch and give identical results (flow_imagenet.pt and rgb_imagenet.pt). These models were pretrained on imagenet and kinetics (see [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) for details). 

## Fine-tuning I3D
[train_i3d.py](train_i3d.py) contains the code to fine-tune I3D based on the details in the paper and obtained from the authors. Specifically, this version follows the settings to fine-tune on the [Charades](allenai.org/plato/charades/) dataset based on the author's implementation that won the Charades 2017 challenge. Our fine-tuned RGB and Flow I3D models are available in the model directory (rgb_charades.pt and flow_charades.pt).

This relied on having the optical flow and RGB frames extracted and saved as images on dist. [charades_dataset.py](charades_dataset.py) contains our code to load video segments for training.


## Feature Extraction
The [demo_extract_features.py](demo_extract_features.py) script loads two entire videos to extract per-segment features. The features of the two videos are in the directory [/output](/output). 
