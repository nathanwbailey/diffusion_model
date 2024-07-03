# Diffusion Model

Implements a Diffusion Model in Keras on the Oxford 102 Flower dataset (https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset). 

Further information can be found in the following blog post:

https://nathanbaileyw.medium.com/a-look-at-diffusion-models-79bd7e789964?postPublishedType=initial

### Code:
The main code is located in the following files:
* main.py - main entry file for training the network
* model.py - implements the diffusion model
* model_building_blocks.py - residual block, Up Block, Down block and sinusoidal embedding to use in the network
* unet_model.py - implements the UNET network for use in the diffusion model
* diffusion_schedules.py - implements various diffusion schedules for training and generating
* image_generator.py - Keras callback to plot images whilst training
* display.py - helper function to plot images
* lint.sh - runs linters on the code
