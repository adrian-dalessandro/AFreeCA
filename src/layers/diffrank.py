import tensorflow as tf

"""
Differential Rank Layer: 
A custom layer that computes the true rank of a sequence. This is a Tensorflow 2.x
implementation of the Pytorch-based Blackbox Combinatorial Solvers (https://github.com/martius-lab/blackbox-backprop)
published by Marin Vlastelica, et al. in the ICLR 2020 paper: "Differentiation of blackbox combinatorial solvers."
"""

def rank_func(seq):
    """
    Compute the rank of each element in a sequence.
    Args:
        seq: A tensor representing the input sequence.
    Returns:
        A tensor with the same shape as the input sequence, containing the ranks of the elements.
    """
    #  Compute the ranks of the elements in the sequence
    return tf.argsort(tf.argsort(seq, axis=-1, direction='DESCENDING'), axis=-1)

def rank_normalised(seq):
    """
    Compute the normalized rank of each element in a sequence.
    Args:
        seq: A tensor representing the input sequence.
    Returns:
        A tensor with the same shape as the input sequence, containing the normalized ranks of the elements.
    """
    # Calculate the normalized ranks of the elements in the sequence
    rank = rank_func(seq)
    rank_plus_one = rank + 1
    seq_size = tf.shape(seq)[1]
    # Normalize the ranks by dividing with the size of the sequence
    return tf.cast(rank_plus_one, dtype=tf.float32) / tf.cast(seq_size, dtype=tf.float32)

@tf.custom_gradient
def true_ranker(sequence, lambda_val):
    """
    Custom autograd function to compute the true rank and gradients.
    Args:
        sequence: A tensor representing the input sequence.
        lambda_val: A float representing the lambda value.
    Returns:
        A tensor with the same shape as the input sequence, containing the true rank values.
        A function representing the gradient of the true rank operation.
    """
    rank = rank_normalised(sequence)
    def grad(grad_output):
        """
        Compute the gradients of the true rank operation.
        Args:
            grad_output: A tensor representing the gradient of the loss with respect to the output.
        Returns:
            A tensor with the same shape as the input sequence, representing the gradients of the input sequence.
        """
        # Calculate the gradients of the sequence based on the rank and gradient of the output
        sequence_prime = sequence + lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        # Compute the gradient for backpropagation using the rank differences
        gradient = -(rank - rank_prime) / (lambda_val + 1e-8)
        return gradient, None
    # Return the rank tensor and the gradient function
    return rank, grad

class TrueRank(tf.keras.layers.Layer):
    """
    Custom Keras layer to apply the true rank operation.
    Args:
        lambda_val: A float representing the lambda value.
    """
    def __init__(self, lambda_val=0.1, **kwargs):
        super(TrueRank, self).__init__(**kwargs)
        self.lambda_val = lambda_val

    def call(self, inputs):
        return true_ranker(inputs, self.lambda_val)

    def get_config(self):
        config = super(TrueRank, self).get_config()
        config.update({"lambda_val": self.lambda_val})
        return config
    