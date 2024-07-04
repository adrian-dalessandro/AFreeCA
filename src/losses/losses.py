from src.utils.similarity import labels_normalized, feature_cosine
import tensorflow as tf

def feature_sort_loss(z_pred, y_true, trank, batch_size):
    y_true_norm = trank(labels_normalized(y_true)) # (batch_size, seq_len, seq_len)
    z_pred_norm = trank(feature_cosine(z_pred)) # (batch_size, seq_len, seq_len)
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.reshape(z_pred_norm, (batch_size, -1)), tf.reshape(y_true_norm, (batch_size, -1))))

def label_sort_loss(y_pred, y_true, trank, batch_size):
    y_pred_norm = trank(labels_normalized(y_pred))
    y_true_norm = trank(labels_normalized(y_true))
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.reshape(y_pred_norm, (batch_size, -1)), tf.reshape(y_true_norm, (batch_size, -1))))

