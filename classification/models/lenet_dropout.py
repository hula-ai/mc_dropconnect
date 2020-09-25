import tensorflow as tf
from models.base_model import BaseModel
from ops import conv_2d, max_pool, flatten_layer, fc_layer, drop_out


class LeNet(BaseModel):
    def __init__(self, sess, conf):
        super(LeNet, self).__init__(sess, conf)
        self.build_network(self.inputs_pl)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('LeNet'):
            x = conv_2d(x, filter_size=5, num_filters=6, name='conv_1', keep_prob=1)
            x = drop_out(x, self.keep_prob_pl)
            x = max_pool(x, 2, 2, 'pool_1')
            x = conv_2d(x, filter_size=5, num_filters=16, name='conv_2', keep_prob=1)
            x = drop_out(x, self.keep_prob_pl)
            x = max_pool(x, 2, 2, 'pool_2')
            x = flatten_layer(x)
            x = drop_out(x, self.keep_prob_pl)
            x = fc_layer(x, 120, name='fc_1', keep_prob=1)
            x = drop_out(x, self.keep_prob_pl)
            x = fc_layer(x, 84, name='fc_2', keep_prob=1)
            x = drop_out(x, self.keep_prob_pl)
            self.logits = fc_layer(x, self.conf.num_cls, name='fc_3', use_relu=False, keep_prob=1)
