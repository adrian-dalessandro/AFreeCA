from src.model.sort.baseline_sort import BaselineSort
from typing import Tuple
import tensorflow as tf
from src.losses.losses import label_sort_loss, feature_sort_loss

"""[RANKSIM SORT CONSISTENCY]-----------------------------------------------------"""

class RankSimSort(BaselineSort):
    def __init__(self, input_shape: Tuple[int, ...], gamma: float, weights: str, augmentor: tf.keras.layers.Layer):
        super(RankSimSort, self).__init__(input_shape, gamma, weights, augmentor)
    
    def compile(self, optimizer: tf.keras.optimizers.Optimizer, f_lambda: float):
        super(RankSimSort, self).compile(optimizer)
        self.f_lambda = f_lambda
        # define metrics to track during training
        self.feat_sort_loss = tf.keras.metrics.Mean(name='feat_sort_loss')

    @property
    def metrics(self):
        return [self.sort_loss, self.feat_sort_loss]
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        encodings = self.encoder(inputs, training=training)
        ouputs = self.decoder(encodings, training=training)
        return ouputs, encodings
    
    def train_step(self, inputs):
        x, y_true = inputs
        # get shape of x, which is (batch, seq_len, H, W, C)
        batch_size, seq_len, H, W, C = x.shape
        # Collapse batch and seq_len dimensions
        x = tf.reshape(x, (-1, H, W, C))
        x = self._preprocess_data(x, True)
        y_true = tf.cast(tf.reshape(y_true, (batch_size, seq_len, 1)), tf.float32)
        with tf.GradientTape() as tape:
            y_pred, z_pred = self(x, training=True)
            y_pred = tf.reshape(y_pred, (batch_size, seq_len, 1))
            z_pred = tf.reshape(z_pred, (batch_size, seq_len, -1))

            # Compute loss
            sort_loss = label_sort_loss(y_pred, y_true, self.trank, batch_size)
            feat_sort_loss = feature_sort_loss(z_pred, y_true, self.trank, batch_size)
            loss = sort_loss + self.f_lambda * feat_sort_loss

        # Compute gradients
        self._apply_gradients(tape, loss)
        # Update metrics
        self.sort_loss.update_state(sort_loss)
        self.feat_sort_loss.update_state(feat_sort_loss)
        return {"sort_loss": self.sort_loss.result(), "feat_sort_loss": self.feat_sort_loss.result()}
    
    @tf.function
    def test_step(self, inputs):
        x, y_true = inputs
        # get shape of x, which is (batch, seq_len, H, W, C)
        x_shape = tf.shape(x)
        batch_size = x_shape[0]
        seq_len = x_shape[1]
        H = x_shape[2]
        W = x_shape[3]
        C = x_shape[4]
        # Collapse batch and seq_len dimensions
        x = tf.reshape(x, (-1, H, W, C))
        x = self._preprocess_data(x, False)
        y_pred, _ = self(x, training=False)
        y_pred = tf.reshape(y_pred, (batch_size, seq_len, 1))
        y_true = tf.cast(tf.reshape(y_true, (batch_size, seq_len, 1)), tf.float32)
        # Compute loss
        loss = label_sort_loss(y_pred, y_true, self.trank, batch_size)
        # Update metrics
        self.sort_loss.update_state(loss)
        return {"sort_loss": self.sort_loss.result()}