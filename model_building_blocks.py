import math
from typing import Any

import tensorflow as tf
from tensorflow import keras


def sinusoidal_embedding(
    x: tf.Tensor, noise_embedding_size: int
) -> tf.Tensor:
    """Sinusoidal Embedding Function."""
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            noise_embedding_size // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


class ResidualBlock(keras.layers.Layer):
    """Residual Block Layer."""

    def __init__(self, width: int, **kwargs: Any) -> None:
        """Init variables and layers."""
        super().__init__()
        self.width = width
        self.downsample_layer = keras.layers.Conv2D(width, kernel_size=1)
        self.batch_norm_layer = keras.layers.BatchNormalization(
            center=False, scale=False
        )
        self.conv_layer_1 = keras.layers.Conv2D(
            width, **kwargs, activation=keras.activations.swish
        )
        self.conv_layer_2 = keras.layers.Conv2D(width, **kwargs)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        if self.width == input.shape[3]:
            residual = input
        else:
            residual = self.downsample_layer(input)
        x = self.batch_norm_layer(input)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        return residual + x


class DownBlock(keras.layers.Layer):
    """Down Block Layer."""

    def __init__(self, block_depth: int, width: int, **kwargs: Any) -> None:
        """Init Layers."""
        super().__init__()
        self.residual_blocks = [
            ResidualBlock(width, **kwargs) for _ in range(block_depth)
        ]
        self.average_pool = keras.layers.AveragePooling2D(pool_size=2)

    def call(self, input: tf.Tensor) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Forward pass."""
        x = input
        skips = []
        for residual in self.residual_blocks:
            x = residual(x)
            skips.append(tf.identity(x))
        x = self.average_pool(x)
        return x, skips


class UpBlock(keras.layers.Layer):
    """Up Block Layer."""

    def __init__(self, block_depth: int, width: int, **kwargs: Any) -> None:
        """Init Layers."""
        super().__init__()
        self.residual_blocks = [
            ResidualBlock(width, **kwargs) for _ in range(block_depth)
        ]
        self.up_sampling = keras.layers.UpSampling2D(
            size=2, interpolation="bilinear"
        )
        self.concat_layer = keras.layers.Concatenate()

    def call(
        self, input_list: list[tf.Tensor | list[tf.Tensor]]
    ) -> tf.Tensor:
        """Forward pass."""
        x, skips = input_list
        x = self.up_sampling(x)
        for residual in self.residual_blocks:
            x = self.concat_layer([x, skips.pop()])
            x = residual(x)
        return x
