import tensorflow as tf


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.contrib.layers.xavier_initializer(uniform=False)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def PReLU(x, scope):
    # PReLU(x) = x if x > 0, alpha*x otherwise

    alpha = tf.get_variable(scope + "/alpha", shape=[1],
                            initializer=tf.constant_initializer(0), dtype=tf.float32)

    output = tf.nn.relu(x) + alpha * (x - abs(x)) * 0.5

    return output


# function for 2D spatial dropout:
def dropout(x, keep_prob):
    # x is a tensor of shape [batch_size, height, width, channels]
    return tf.nn.dropout(x, keep_prob)


# function for unpooling max_pool:
def max_unpool(inputs, pooling_indices, output_shape=None, k_size=[1, 2, 2, 1]):
    # NOTE! this function is based on the implementation by kwotsin in
    # https://github.com/kwotsin/TensorFlow-ENet

    # inputs has shape [batch_size, height, width, channels]

    # pooling_indices: pooling indices of the previously max_pooled layer

    # output_shape: what shape the returned tensor should have

    pooling_indices = tf.cast(pooling_indices, tf.int32)
    input_shape = tf.shape(inputs, out_type=tf.int32)

    one_like_pooling_indices = tf.ones_like(pooling_indices, dtype=tf.int32)
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
    batch_range = tf.reshape(tf.range(input_shape[0], dtype=tf.int32), shape=batch_shape)
    b = one_like_pooling_indices * batch_range
    y = pooling_indices // (output_shape[2] * output_shape[3])
    x = (pooling_indices // output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype=tf.int32)
    f = one_like_pooling_indices * feature_range

    inputs_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, inputs_size]))
    values = tf.reshape(inputs, [inputs_size])

    ret = tf.scatter_nd(indices, values, output_shape)

    return ret


def drop_connect(w, keep_prob):
    return tf.nn.dropout(w, keep_prob=keep_prob) * keep_prob
