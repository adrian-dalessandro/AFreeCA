import tensorflow as tf

"""
A keras layer for preprocessing the data for a pretrained neural network using the preprocessing scripts under
tf.keras.applications.
Works for resnet50_v2 and vgg16.
"""
class Preprocess(tf.keras.layers.Layer):
    def __init__(self, model_name):
        super(Preprocess, self).__init__()
        self.model_name = model_name
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_name": self.model_name
            }
        )
        return config

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, 0, 1)
        if self.model_name == 'resnet50':
            return tf.keras.applications.resnet_v2.preprocess_input(255*inputs)
        elif self.model_name == 'vgg16':
            return tf.keras.applications.vgg16.preprocess_input(255*inputs)
        else:
            raise ValueError("Invalid model name. Choose from resnet50 or vgg16.")