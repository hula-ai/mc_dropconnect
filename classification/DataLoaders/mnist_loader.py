import random
import scipy
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.mnist = input_data.read_data_sets("data/mnist", one_hot=True)

    def get_data(self, mode='train'):
        if mode == 'train':
            x_train, self.y_train = self.mnist.train.images, self.mnist.train.labels
            self.x_train = x_train.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.channel))
        elif mode == 'valid':
            x_valid, self.y_valid = self.mnist.validation.images, self.mnist.validation.labels
            self.x_valid = x_valid.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.channel))
        elif mode == 'test':
            x_test, self.y_test = self.mnist.test.images, self.mnist.test.labels
            self.x_test = x_test.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.channel))

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            x, y = self.mnist.train.next_batch(self.cfg.batch_size)
            x = x.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.channel))
            if self.augment:
                x = random_rotation_2d(x, self.cfg.max_angle)
        elif mode == 'valid':
            x = self.x_valid[start:end]
            y = self.y_valid[start:end]
        elif mode == 'test':
            x = self.x_test[start:end]
            y = self.y_test[start:end]
        return x, y

    def count_num_batch(self, batch_size, mode='train'):
        if mode == 'train':
            num_batch = int(self.y_train.shape[0] / batch_size)
        elif mode == 'valid':
            num_batch = int(self.y_valid.shape[0] / batch_size)
        elif mode == 'test':
            num_batch = int(self.y_test.shape[0] / batch_size)
        return num_batch

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        shuffled_x = self.x_train[permutation, :, :, :]
        shuffled_y = self.y_train[permutation]
        return shuffled_x, shuffled_y


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)
