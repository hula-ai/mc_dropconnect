import random
import numpy as np
import h5py
import scipy.ndimage
import glob


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.train_data_dir = cfg.train_data_dir
        self.valid_data_dir = cfg.valid_data_dir
        self.test_data_dir = cfg.test_data_dir
        self.batch_size = cfg.batch_size
        self.num_tr = cfg.num_tr
        self.height, self.width, self.depth, self.channel = cfg.height, cfg.width, cfg.depth, cfg.channel
        project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
        self.train_files = glob.glob(project_path + self.cfg.train_data_dir + '*.h5')
        self.valid_files = glob.glob(project_path + self.cfg.valid_data_dir + '*.h5')
        self.test_files = glob.glob(project_path + self.cfg.test_data_dir + '*.h5')

    def next_batch(self, num=None, mode='train'):
        if mode == 'train':
            img_idx = np.sort(np.random.choice(self.num_tr, replace=False, size=self.batch_size))
            x, y = self.random_crop(img_idx)
            if self.augment:
                x, y = random_rotation_3d(x, y, max_angle=self.max_angle)
        elif mode == 'valid':
            h5f = h5py.File(self.valid_files[num], 'r')
            x = h5f['x_norm'][:]
            y = h5f['y'][:]
            h5f.close()
            Dcuts = [num for num in range(0, x.shape[-2], self.cfg.Dcut_size)][1:]
            x = np.concatenate(np.split(x, Dcuts, axis=-2)[:-1], axis=0)
            y = np.concatenate(np.split(y, Dcuts, axis=-1)[:-1], axis=0)
        elif mode == 'test':
            h5f = h5py.File(self.test_files[num], 'r')
            x = h5f['x_norm'][:]
            y = h5f['y'][:]
            h5f.close()
            Dcuts = [num for num in range(0, x.shape[-2], self.cfg.Dcut_size)][1:]
            x = np.concatenate(np.split(x, Dcuts, axis=-2)[:-1], axis=0)
            y = np.concatenate(np.split(y, Dcuts, axis=-1)[:-1], axis=0)
        return x, y

    def count_num_samples(self, mode='valid'):
        if mode == 'valid':
            num_ = len(self.valid_files)
        elif mode == 'test':
            num_ = len(self.valid_files)
        return num_

    def random_crop(self, img_idx):
        if self.cfg.random_crop:
            x = np.zeros([self.batch_size] + self.cfg.crop_size + [self.channel])
            y = np.zeros([self.batch_size] + self.cfg.crop_size)
            height, width, depth = self.cfg.crop_size
            for i, num in enumerate(img_idx):
                h5f = h5py.File(self.train_files[num], 'r')
                scan_depth = h5f['x_norm'][:].shape[-2]
                HStart = np.random.randint(low=0, high=512 - height - 1)
                WStart = np.random.randint(low=0, high=512 - width - 1)
                DStart = np.random.randint(low=0, high=scan_depth - depth - 1)
                x[i] = h5f['x_norm'][:, HStart:HStart+height, WStart:WStart+width, DStart:DStart+depth]
                y[i] = h5f['y'][:, HStart:HStart+height, WStart:WStart+width, DStart:DStart + depth]
                h5f.close()
        else:
            x = np.zeros((self.batch_size, self.height, self.width, self.depth, self.channel))
            y = np.zeros((self.batch_size, self.height, self.width, self.depth))
            for i, num in enumerate(img_idx):
                h5f = h5py.File(self.train_files[num], 'r')
                scan_depth = h5f['x_norm'][:].shape[-2]
                DStart = np.random.randint(low=0, high=scan_depth - self.depth - 1)
                x[i] = h5f['x_norm'][:, :, :, DStart:DStart + self.depth]
                y[i] = h5f['y'][:, :, :, DStart:DStart + self.depth]
                h5f.close()
        return x, y


def random_rotation_3d(img_batch, mask_batch, max_angle):
    """
    Randomly rotate an image by a random angle (-max_angle, max_angle)
    :param img_batch: batch of 3D images
    :param mask_batch: batch of 3D masks
    :param max_angle: `float`. The maximum rotation angle
    :return: batch of rotated 3D images and masks
    """
    size = img_batch.shape
    img_batch = np.squeeze(img_batch, axis=-1)
    img_batch_rot, mask_batch_rot = img_batch, mask_batch
    for i in range(img_batch.shape[0]):
        axis_rot = np.random.randint(2, size=3)
        if np.sum(axis_rot):    # if rotating along any axis
            image, mask = img_batch[i], mask_batch[i]
            if axis_rot[0]:
                # rotate along z-axis
                angle = random.uniform(-max_angle, max_angle)
                image = rotate(image, angle)
                mask = rotate(mask, angle)
            if axis_rot[1]:
                # rotate along y-axis
                angle = random.uniform(-max_angle, max_angle)
                image = rotate(image, angle)
                mask = rotate(mask, angle)
            if axis_rot[2]:
                # rotate along x-axis
                angle = random.uniform(-max_angle, max_angle)
                image = rotate(image, angle)
                mask = rotate(mask, angle)
            img_batch_rot[i] = image
            mask_batch_rot[i] = mask
    return img_batch_rot.reshape(size), mask_batch


def rotate(x, angle):
    return scipy.ndimage.interpolation.rotate(x, angle, mode='nearest', axes=(0, 1), reshape=False)
