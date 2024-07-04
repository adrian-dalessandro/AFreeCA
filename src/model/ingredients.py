import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.applications import ResNet50


class EncoderModel(tf.keras.Model):
    def __init__(self, input_shape, weights):
        super(EncoderModel, self).__init__()
        self.base_model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
        self.global_average_pooling = GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_average_pooling(x)
        return x

    def get_config(self):
        config = super(EncoderModel, self).get_config()
        config.update({
            'input_shape': self.base_model.input_shape,
            'weights': self.base_model.weights,
        })
        return config

class DecoderModel(tf.keras.Model):
    def __init__(self):
        super(DecoderModel, self).__init__()
        self.dense_layer = Dense(1, activation=None, kernel_constraint=non_neg())
        self.activation = Activation('relu', dtype='float32')

    def call(self, inputs):
        x = self.dense_layer(inputs)
        predictions = self.activation(x)
        return predictions

    def get_config(self):
        config = super(DecoderModel, self).get_config()
        return config
