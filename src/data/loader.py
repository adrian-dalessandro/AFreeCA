from src.data.helper import get_triplet_image_ds
from tensorflow.data import AUTOTUNE
from typing import List, Tuple
import tensorflow as tf
import numpy as np

class TripletDataIterator(object):
    """
    Generates a data iterator of a triplet of images with an ordering. The iterator has the structure ((im1, im2, im3), [0, 1, 2])
    Args:
        dataset: list with three image sorted paths, which are the relative paths to the image files within a directory
        path: path to the directory containing the image files
        size: size of the images
    """
    def __init__(
            self, 
            dataset: List[dict], 
            path: str, 
            size: Tuple[int, ...]):
        self.first = [path + d[0] for d in dataset]
        self.second = [path + d[1] for d in dataset]
        self.third = [path + d[2] for d in dataset]
        self.size = size
        
    def build(
            self,  
            batch_size: int, 
            drop_remainder: bool, 
            shuffle: bool) -> tf.data.Dataset:
        rank_ds = get_triplet_image_ds(self.first, self.second, self.third, self.size, shuffle).batch(batch_size, drop_remainder=drop_remainder)
        return rank_ds.prefetch(AUTOTUNE)