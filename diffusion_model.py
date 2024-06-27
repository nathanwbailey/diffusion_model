import tensorflow as tf
from tensorflow import keras
from unet_model import create_unet_model
from diffusion_schedules import offset_cosine_diffusion_schedule

class DiffusionModel(keras.models.Model):
    def __init__(self, image_size: int, batch_size: int, ema_value: float, noise_embedding_size: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.ema_value = ema_value
        self.normalizer = keras.layers.Normalization()
        self.network = create_unet_model(image_size, 2, [32, 64, 96, 128], noise_embedding_size)
        self.ema_network = keras.models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
    
    @property
    def metrics(self) -> keras.metrics.Metric:
        return [self.noise_loss_tracker]

    def denormalize(self, images: tf.Tensor) -> tf.Tensor:
        images = self.normalizer.mean + images * self.normalizer.variance **0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images: tf.Tensor, noise_rates: tf.Tensor, signal_rates: tf.Tensor, training: bool = True) -> tuple[tf.Tensor, tf.Tensor]:
        if training:
            network = self.network
        else:
            network = self.ema_network
        
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates*pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise: tf.Tensor, diffusion_steps: int) -> tf.Tensor:
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        # Timesteps are float values between 0 and 1
        # Start from final step t (1) and work backwards
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
        return pred_images

    def generate(self, num_images: int, diffusion_steps: int, initial_noise: tf.Tensor | None = None) -> tf.Tensor:
        if initial_noise is None:
            initial_noise = tf.random.normal(shape=(num_images, self.image_size, self.image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images: tf.Tensor) -> dict[str, tf.Tensor]:
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 3))

        #Generate an image x(t) at a random timestep t
        diffusion_times = tf.random.uniform(shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates)
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema_value*ema_weight + (1-self.ema_value)* weight)
        return {m.name: m.result() for m in self.metrics}