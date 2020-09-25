import tensorflow as tf


def get_num_channels(x):
    """
    returns the input's number of channels
    :param x: input tensor with shape [batch_size, ..., num_channels]
    :return: number of channels
    """
    return x.get_shape().as_list()[-1]


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


def conv_2d(inputs, filter_size, num_filters, layer_name, add_batch_norm, is_train,
            stride=1, add_reg=True, activation=tf.identity, keep_prob=None):
    """
    Create a 2D convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param add_batch_norm: boolean to use batch norm (or not)
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param activation: type of activation to be applied
    :return: The output array
    """
    num_in_channel = get_num_channels(inputs)
    with tf.variable_scope(layer_name):
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        weights = tf.reshape(drop_connect(weights, keep_prob), shape=shape)
        tf.summary.histogram('W', weights)
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        print('{}: {}'.format(layer_name, layer.get_shape()))
        if add_batch_norm:
            layer = batch_norm(layer, is_train, layer_name)
        else:
            biases = bias_variable(layer_name, [num_filters])
            layer += biases
        layer = activation(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def deconv_2d(inputs, filter_size, num_filters, layer_name, stride=1, add_batch_norm=False,
              is_train=True, add_reg=False, activation=tf.identity):
    """
    Create a 2D transposed-convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param batch_norm: boolean to use batch norm (or not)
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param activation: type of activation to be applied
    :param out_shape: Tensor of output shape
    :return: The output array
    """
    with tf.variable_scope(layer_name):
        layer = tf.layers.conv2d_transpose(inputs,
                                           filters=num_filters,
                                           kernel_size=[filter_size, filter_size],
                                           strides=[stride, stride],
                                           padding="SAME",
                                           use_bias=False)
        print('{}: {}'.format(layer_name, layer.get_shape()))
        if add_batch_norm:
            layer = batch_norm(layer, is_train, layer_name)
        else:
            biases = bias_variable(layer_name, [num_filters])
            layer += biases
        layer = activation(layer)
        # if add_reg:
        #    tf.add_to_collection('weights', weights)
    return layer


def BN_Relu_conv_2d(inputs, filter_size, num_filters, layer_name, stride=1, is_train=True,
                    add_batch_norm=True, use_relu=True, add_reg=False):
    """
    Create a BN, ReLU, and 2D convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param add_batch_norm: boolean to use batch norm (or not)
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu:
    :return: The output array
    """
    num_in_channel = get_num_channels(inputs)
    with tf.variable_scope(layer_name):
        if add_batch_norm:
            inputs = batch_norm(inputs, is_train, layer_name)
        if use_relu:
            inputs = tf.nn.relu(inputs)
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def max_pool(x, ksize, stride, name):
    """
    Create a 3D max-pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    maxpool = tf.nn.max_pool(x,
                             ksize=[1, ksize, ksize, 1],
                             strides=[1, stride, stride, 1],
                             padding="SAME",
                             name=name)
    print('{}: {}'.format(name, maxpool.get_shape()))
    return maxpool


def avg_pool(x, ksize, stride, scope):
    """Create an average pooling layer."""
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="VALID",
                          name=scope)


def batch_norm(bias_input, is_training, scope):
    with tf.variable_scope(scope):
        return tf.cond(is_training,
                       lambda: tf.contrib.layers.batch_norm(bias_input, is_training=True, center=False, scope=scope),
                       lambda: tf.contrib.layers.batch_norm(bias_input, is_training=False, center=False, reuse=True,
                                                            scope=scope))


# def batch_norm(inputs, is_training, scope='BN', decay=0.999, epsilon=1e-3):
#     """
#     creates a batch normalization layer
#     :param inputs: input array
#     :param is_training: boolean for differentiating train and test
#     :param scope: scope name
#     :param decay:
#     :param epsilon:
#     :return: normalized input
#     """
#     with tf.variable_scope(scope):
#         scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
#         beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
#         pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
#         pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
#
#         if is_training:
#             if len(inputs.get_shape().as_list()) == 5:  # For 3D convolutional layers
#                 batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2, 3])
#             else:  # For fully-connected layers
#                 batch_mean, batch_var = tf.nn.moments(inputs, [0])
#             train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
#             train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
#             with tf.control_dependencies([train_mean, train_var]):
#                 return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
#         else:
#             return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def prelu(x, name=None):
    """
    Applies parametric leaky ReLU
    :param x: input tensor
    :param name: variable name
    :return: output tensor of the same shape
    """
    with tf.variable_scope(name_or_scope=name, default_name="prelu"):
        alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype,
                                initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def Relu(x):
    return tf.nn.relu(x)


#
# def drop_out(x, keep_prob):
#     return tf.nn.dropout(x, keep_prob)


def concatenation(layers):
    return tf.concat(layers, axis=-1)


def drop_connect(w, keep_prob):
    return tf.nn.dropout(w, keep_prob=keep_prob) * keep_prob
