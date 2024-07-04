import tensorflow as tf
from tensorflow.data import Dataset
    

@tf.function
def parse_images(image_path, size):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[size[0], size[1]])
    #image = tf.image.resize_with_pad(image, size[0], size[1])
    return image

def get_triplet_image_ds(first, second, third, size, shuffle=False):
    """
    get_paired_image_ds is a function that takes two lists of image paths and 
    returns a dataset of paired images with a label of 1
    """
    def func(x, y, z, size):
        images = [parse_images(x, size)[None], parse_images(y, size)[None], parse_images(z, size)[None]]
        return tf.concat(images, axis=0)
    first_ds = Dataset.from_tensor_slices(first)
    second_ds = Dataset.from_tensor_slices(second)
    third_ds = Dataset.from_tensor_slices(third)
    triplet_ds = Dataset.zip((first_ds, second_ds, third_ds))
    if shuffle:
        triplet_ds = triplet_ds.shuffle(6000)
    triplet_ds = triplet_ds.map(lambda x, y, z: (func(x, y, z, size), [0, 1, 2]), num_parallel_calls=tf.data.AUTOTUNE)
    return triplet_ds