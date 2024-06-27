import numpy as np
import tensorflow as tf
from tensorflow import keras
from diffusion_model import DiffusionModel
from image_generator import ImageGenerator
from display import display

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5
NOISE_EMBEDDING_SIZE = 32

EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
NUM_DIFFUSION_STEPS=20

train_data = keras.utils.image_dataset_from_directory(
    "data/flower-dataset/dataset",
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

model = DiffusionModel(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, ema_value=EMA, noise_embedding_size=NOISE_EMBEDDING_SIZE)
model.normalizer.adapt(train_data)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY), loss=keras.losses.mean_absolute_error)
model.network.summary(expand_nested=True)

image_generator = ImageGenerator(num_img=100, num_diffusion_steps=NUM_DIFFUSION_STEPS)
model.fit(train_data, epochs=EPOCHS, callbacks=[image_generator], verbose=2)

generated_images = model.generate(num_images=100, diffusion_steps=NUM_DIFFUSION_STEPS).numpy()

display(generated_images, save_to='final_generated_images.png')