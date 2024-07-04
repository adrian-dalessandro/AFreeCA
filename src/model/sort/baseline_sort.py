from typing import Tuple
import tensorflow as tf
from src.model.ingredients import EncoderModel, DecoderModel
from src.layers.preprocess import Preprocess
from src.layers.diffrank import TrueRank
from src.losses.losses import label_sort_loss

class BaselineSort(tf.keras.Model):
    def __init__(self, input_shape: Tuple[int, ...], gamma: float, weights: str, augmentor: tf.keras.layers.Layer):
        super(BaselineSort, self).__init__()
        self.encoder = EncoderModel(input_shape, weights)
        self.decoder = DecoderModel()
        self.augmentor = augmentor
        self.preprocess = Preprocess("resnet50")
        self.trank = TrueRank(gamma)
        self.model_list = {"encoder": self.encoder, "decoder": self.decoder}

    def compile(self, optimizer):
        super(BaselineSort, self).compile()
        self.optimizer = optimizer
        # define metrics to track during training
        self.sort_loss = tf.keras.metrics.Mean(name='sort_loss')
    
    @property
    def metrics(self):
        return [self.sort_loss]
    
    def _preprocess_data(self, x, augment):
        if augment:
            x = self.preprocess(self.augmentor(x, training=True))
        else:
            x = self.preprocess(x)
        return tf.concat(x, axis=0)
    
    def _apply_gradients(self, tape, loss):
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        encodings = self.encoder(inputs, training=training)
        ouputs = self.decoder(encodings, training=training)
        return ouputs

    def train_step(self, inputs):
        x, y_true = inputs
        # get shape of x, which is (batch, seq_len, H, W, C)
        batch_size, seq_len, H, W, C = x.shape
        # Collapse batch and seq_len dimensions
        x = tf.reshape(x, (-1, H, W, C))
        x = self._preprocess_data(x, True)
        with tf.GradientTape() as tape:
            y_true = tf.cast(tf.reshape(y_true, (batch_size, seq_len, 1)), tf.float32)
            y_pred = self(x, training=True)
            y_pred = tf.reshape(y_pred, (batch_size, seq_len, 1))

            loss = label_sort_loss(y_pred, y_true, self.trank, batch_size)

        # Compute gradients
        self._apply_gradients(tape, loss)
        # Update metrics
        self.sort_loss.update_state(loss)
        return {"sort_loss": self.sort_loss.result()}
    
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
        y_pred = self(x, training=False)
        y_pred = tf.reshape(y_pred, (batch_size, seq_len, 1))
        y_true = tf.cast(tf.reshape(y_true, (batch_size, seq_len, 1)), tf.float32)
        # Compute loss
        loss = label_sort_loss(y_pred, y_true, self.trank, batch_size)
        # Update metrics
        self.sort_loss.update_state(loss)
        return {"sort_loss": self.sort_loss.result()}
    
    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """
        Overwrite the save method with a custom method which savs ONLY the encoder model
        """
        for name, model in self.model_list.items():
            model.save(filepath + "/" + name, overwrite=overwrite, save_format=save_format, **kwargs)