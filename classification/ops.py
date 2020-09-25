import tensorflow as tf


def get_num_channels(x):
    return x.get_shape().as_list()[-1]


def weight_variable(name, shape):
    initer = tf.contrib.layers.xavier_initializer(uniform=False)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def conv_2d(inputs, filter_size, num_filters, name, stride=1, add_reg=True, add_relu=True, keep_prob=None):
    num_in_channel = get_num_channels(inputs)
    with tf.variable_scope(name):
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(name, shape=shape)
        weights = tf.reshape(drop_connect(weights, keep_prob), shape=shape)
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        print('{}: {}'.format(name, layer.get_shape()))
        biases = bias_variable(name, [num_filters])
        layer += biases
        if add_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def max_pool(x, ksize, stride, name):
    maxpool = tf.nn.max_pool(x,
                             ksize=[1, ksize, ksize, 1],
                             strides=[1, stride, stride, 1],
                             padding="SAME",
                             name=name)
    print('{}: {}'.format(name, maxpool.get_shape()))
    return maxpool


def flatten_layer(layer):
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def fc_layer(bottom, out_dim, name, add_reg=True, use_relu=True, keep_prob=None):
    in_dim = bottom.get_shape()[1]
    with tf.variable_scope(name):
        weights = weight_variable(name, shape=[in_dim, out_dim])
        weights = tf.reshape(drop_connect(weights, keep_prob), shape=[in_dim, out_dim])
        biases = bias_variable(name, [out_dim])
        layer = tf.matmul(bottom, weights)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def drop_out(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def drop_connect(w, keep_prob):
    return tf.nn.dropout(w, keep_prob=keep_prob) * keep_prob


def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)
