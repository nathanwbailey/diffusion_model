import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5
NOISE_EMBEDDING_SIZE = 32

EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50


train_data = keras.utils.image_dataset_from_directory(
    "../data/flower-dataset/dataset",
    labels=None,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=None,
    shuffle=True,
    seed=42,
    interpolation="bilinear"
)

def preprocess_images(img):
    img = tf.cast(img, "float32") / 255.0
    return img

train_data = train_data.map(preprocess_images)
train_data = train_data.repeat(DATASET_REPETITIONS)
train_data = train_data.batch(BATCH_SIZE, drop_remainder=True)


