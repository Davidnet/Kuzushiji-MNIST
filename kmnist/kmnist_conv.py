import tensorflow as tf

def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])