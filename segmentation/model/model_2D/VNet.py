import tensorflow as tf
from model.model_2D.base_model import BaseModel
from model.model_2D.ops import conv_2d, deconv_2d, prelu, drop_connect
from model.model_2D.ops import get_num_channels


class VNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=6,
                 num_convs=(2, 3, 3, 4, 4, 4),
                 bottom_convs=5,
                 act_fcn=prelu):

        super(VNet, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.bottom_convs = bottom_convs
        self.k_size = self.conf.filter_size
        self.down_conv_factor = 2
        self.act_fcn = act_fcn
        self.build_network(self.inputs_pl)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('VNet'):
            feature_list = list()

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        x = self.conv_block_down(x, self.num_convs[l])
                        feature_list.append(x)
                        x = self.down_conv(x)

            with tf.variable_scope('Bottom_level'):
                x = self.conv_block_down(x, self.bottom_convs)

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        f = feature_list[l]
                        x = self.up_conv(x)
                        x = self.conv_block_up(x, f, self.num_convs[l])

            self.logits = conv_2d(x, 1, self.conf.num_cls, 'Output_layer', self.conf.use_BN,
                                  self.is_training_pl, keep_prob=1)

    def conv_block_down(self, layer_input, num_convolutions):
        x = layer_input
        n_channels = get_num_channels(x)
        if n_channels == 1:
            n_channels = self.conf.start_channel_num
        for i in range(num_convolutions):
            x = conv_2d(inputs=x,
                        filter_size=self.k_size,
                        num_filters=n_channels,
                        layer_name='conv_' + str(i + 1),
                        add_batch_norm=self.conf.use_BN,
                        is_train=self.is_training_pl,
                        keep_prob=self.keep_prob_pl)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = self.act_fcn(x, name='prelu_' + str(i + 1))
            # x = tf.layers.dropout(x, rate=(1 - self.keep_prob_pl), training=self.with_dropout_p)
            # x = tf.nn.dropout(x, keep_prob=self.keep_prob_pl)
        return x

    def conv_block_up(self, layer_input, fine_grained_features, num_convolutions):
        x = tf.concat((layer_input, fine_grained_features), axis=-1)
        n_channels = get_num_channels(layer_input)
        for i in range(num_convolutions):
            x = conv_2d(inputs=x,
                        filter_size=self.k_size,
                        num_filters=n_channels,
                        layer_name='conv_' + str(i + 1),
                        add_batch_norm=self.conf.use_BN,
                        is_train=self.is_training_pl,
                        keep_prob=self.keep_prob_pl)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = self.act_fcn(x, name='prelu_' + str(i + 1))
            # x = tf.layers.dropout(x, rate=(1 - self.keep_prob_pl), training=self.with_dropout_pl)
            # x = tf.nn.dropout(x, keep_prob=self.keep_prob_pl)
        return x

    def down_conv(self, x):
        num_out_channels = get_num_channels(x) * 2
        x = conv_2d(inputs=x,
                    filter_size=2,
                    num_filters=num_out_channels,
                    layer_name='conv_down',
                    stride=2,
                    add_batch_norm=self.conf.use_BN,
                    is_train=self.is_training_pl,
                    keep_prob=self.keep_prob_pl,
                    activation=self.act_fcn)
        return x

    def up_conv(self, x):
        num_out_channels = get_num_channels(x) // 2
        x = deconv_2d(inputs=x,
                      filter_size=2,
                      num_filters=num_out_channels,
                      layer_name='conv_up',
                      stride=2,
                      add_batch_norm=self.conf.use_BN,
                      is_train=self.is_training_pl)
        return x
