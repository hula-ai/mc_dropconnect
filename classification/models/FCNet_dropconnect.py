import tensorflow as tf
from models.base_model import BaseModel
from ops import conv_2d, max_pool, flatten_layer, fc_layer, drop_out


class FCNet(BaseModel):
    def __init__(self, sess, conf):
        super(FCNet, self).__init__(sess, conf)
        self.build_network(self.inputs_pl)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('FCNet'):
            x = conv_2d(x, filter_size=3, stride=1, num_filters=32, name='conv_1', keep_prob=self.conf.keep_prob)
            x = tf.contrib.slim.batch_norm(x)
            x = conv_2d(x, filter_size=3, stride=1, num_filters=32, name='conv_2', keep_prob=self.conf.keep_prob)
            x = tf.contrib.slim.batch_norm(x)
            x = max_pool(x, 2, 2, 'pool_1')

            x = conv_2d(x, filter_size=3, stride=1, num_filters=64, name='conv_3', keep_prob=self.conf.keep_prob)
            x = tf.contrib.slim.batch_norm(x)
            x = conv_2d(x, filter_size=3, stride=1, num_filters=64, name='conv_4', keep_prob=self.conf.keep_prob)
            x = tf.contrib.slim.batch_norm(x)
            x = max_pool(x, 2, 2, 'pool_2')

            x = conv_2d(x, filter_size=3, stride=1, num_filters=128, name='conv_5', keep_prob=self.conf.keep_prob)
            x = tf.contrib.slim.batch_norm(x)
            x = conv_2d(x, filter_size=3, stride=1, num_filters=128, name='conv_6', keep_prob=self.conf.keep_prob)
            x = tf.contrib.slim.batch_norm(x)
            x = max_pool(x, 2, 2, 'pool_3')

            x = flatten_layer(x)
            self.logits = fc_layer(x, self.conf.num_cls, name='fc_3', use_relu=False, keep_prob=1)

