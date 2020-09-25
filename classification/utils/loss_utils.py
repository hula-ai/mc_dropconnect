import tensorflow as tf


def cross_entropy(labels_tensor, logits_tensor):
    diff = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_tensor, labels=labels_tensor)
    loss = tf.reduce_mean(diff)
    return loss

