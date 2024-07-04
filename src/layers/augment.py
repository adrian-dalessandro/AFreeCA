import tensorflow as tf
from tensorflow.keras import layers
from src.layers.preprocess import Preprocess

"""
A script for composing several augmentation strategies into one model:
1. Randomly flip the image horizontally
2. Randomly zoom out (by a factor of 1.0 to 1.5)
3. Random gaussian blur
4. Random Color Affine
5. Random brightness
6. Random contrast
"""

class RandomImageNoise(layers.Layer):
    """
    RandomImageNoise is a class that applies a random noise to an image.
    Each application applies a shot noise with a random sigma between 0 and sigma.
    This noise is applied to the whole image.
    intensity: the intensity of the image noise
    """
    def __init__(self, intensity, **kwargs):
        super().__init__(**kwargs)
        self.intensity = intensity
    
    def get_config(self):
        config = super().get_config()
        config.update({"intensity": self.intensity})
        return config

    @tf.function
    def call(self, images, training=True):
        if training:
            noise = tf.random.uniform(tf.shape(images), minval=-self.intensity, maxval=self.intensity)
            images = tf.clip_by_value(images + noise, 0, 1)
        return images


class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    @tf.function
    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images
    

class AugmentModel(layers.Layer):
    """
    AugmentModel is a class that takes in a base model and applies several augmentation strategies to the input image.
    The augmented image is then passed through the base model, and the output is returned.
    The augmentation strategies are:
    1. Randomly flip the image horizontally
    2. Randomly zoom out (by a factor of 1.0 to 1.5)
    4. Random Color Affine
    5. Random brightness
    6. Random contrast
    """
    def __init__(self, zoom_args, blur_args, color_args, contrast_args, noise_args, model_args, **kwargs):
        super().__init__(**kwargs)
        self.zoom_args = zoom_args
        self.blur_args = blur_args
        self.color_args = color_args
        self.contrast_args = contrast_args
        self.noise_args = noise_args
        self.model_args = model_args

        self.flip = layers.RandomFlip("horizontal")
        self.zoom = layers.RandomZoom(**self.zoom_args)
        self.color = RandomColorAffine(**self.color_args)
        self.contrast = layers.RandomContrast(**self.contrast_args)
        self.noise = RandomImageNoise(**self.noise_args)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "zoom_args": self.zoom_args,
                "blur_args": self.blur_args,
                "color_args": self.color_args,
                "contrast_args": self.contrast_args,
                "noise_args": self.noise_args,
                "model_args": self.model_args
            }
        )
        return config

    def call(self, images, training=True):
        if training:
            images = self.flip(images)
            images = self.zoom(images)
            images = self.color(images)
            images = self.contrast(images)
            images = self.noise(images)
        return images