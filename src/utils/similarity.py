import tensorflow as tf

def feature_cosine(feats):
    feats_normalized = tf.math.l2_normalize(feats, axis=-1)
    return tf.matmul(feats_normalized, tf.transpose(feats_normalized, [0, 2, 1]))

def labels_normalized(labels):
    return -tf.abs(tf.transpose(labels, [0, 2, 1]) - labels)