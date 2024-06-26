from tensorflow import keras

from display import display


class ImageGenerator(keras.callbacks.Callback):
    """Image Generator Callback."""

    def __init__(self, num_img: int, num_diffusion_steps: int) -> None:
        """Init variables."""
        super().__init__()
        self.num_img = num_img
        self.num_diffusion_steps = num_diffusion_steps

    def on_epoch_end(self, epoch: int, logs: None = None) -> None:
        """Generate and save images on epoch end."""
        generated_images = self.model.generate(
            num_images=self.num_img, diffusion_steps=self.num_diffusion_steps
        ).numpy()
        display(
            generated_images,
            save_to=f"./output/generated_image_epoch_{epoch}.png",
        )
