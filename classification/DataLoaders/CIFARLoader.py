import random
import scipy
import numpy as np
import h5py
from keras.utils.np_utils import to_categorical


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.batch_size = cfg.batch_size
        if cfg.height == 32:
            self.data_path = './classification/data/cifar-10-conv.h5'
        elif cfg.height == 96:
            self.data_path = './classification/data/cifar-10-96.h5'
            
    def get_data(self, mode='train'):
        if mode == 'train':
            h5f = h5py.File(self.data_path, 'r')
            x_train = h5f['x_train'][:]
            y_train = h5f['y_train'][:]
            h5f.close()
            self.x_train = x_train
            self.y_train = to_categorical(y_train, num_classes=10)

        elif mode == 'valid':
            h5f = h5py.File(self.data_path, 'r')
            x_valid = h5f['x_test'][:]
            y_valid = h5f['y_test'][:]
            h5f.close()
            self.x_valid = x_valid
            self.y_valid = to_categorical(y_valid, num_classes=10)
        elif mode == 'test':
            h5f = h5py.File(self.data_path, 'r')
            x_test = h5f['x_test'][:]
            y_test = h5f['y_test'][:]
            h5f.close()
            self.x_test = x_test
            self.y_test = to_categorical(y_test, num_classes=10)

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            x = self.x_train[start:end]
            y = self.y_train[start:end]
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
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation]


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

        # horizontal flip
        if bool(random.getrandbits(1)):
            batch_rot[i] = np.fliplr(batch_rot[i])
    return batch_rot.reshape(size)
