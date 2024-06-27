import tensorflow as tf
from tensorflow import keras
from unet_model import create_unet_model

class DiffusionModel(keras.models.Model):
    def __init__(self):
        