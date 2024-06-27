from tensorflow import keras
from model_building_blocks import sinusoidal_embedding
from model_building_blocks import ResidualBlock
from model_building_blocks import UpBlock
from model_building_blocks import DownBlock

def create_unet_model(image_size: int, block_depth: int, filter_list: list[int], noise_embedding_size: int) -> keras.models.Model:

    assert len(filter_list) == 4

    noisy_images = keras.layers.Input(shape=(image_size, image_size, 3))
    x = keras.layers.Conv2D(32, kernel_size=1)(noisy_images)

    noise_variances = keras.layers.Input(shape=(1,1,1))
    noise_embedding = keras.layers.Lambda(sinusoidal_embedding, arguments={"noise_embedding_size": noise_embedding_size})(noise_variances)
    noise_embedding = keras.layers.UpSampling2D(size=image_size, interpolation="nearest")(noise_embedding)

    x = keras.layers.Concatenate()([x, noise_embedding])

    skips_total = []
    for filter_width in filter_list[:-1]:
        x, skips = DownBlock(block_depth=block_depth, width=filter_width, kernel_size=3, padding="same")(x)
        skips_total += skips
    
    for _ in range(2):
        x = ResidualBlock(width=filter_list[-1], kernel_size=3, padding="same")(x)
    for filter_width in filter_list[:-1][::-1]:
        x = UpBlock(block_depth=block_depth, width=filter_width, kernel_size=3, padding="same")([x, [skips_total.pop(), skips_total.pop()]])
        
    
    x = keras.layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.models.Model([noisy_images, noise_variances], x)

    