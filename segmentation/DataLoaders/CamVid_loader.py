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
        self.batch_size = cfg.batch_size
        self.height, self.width, self.depth, self.channel = cfg.height, cfg.width, cfg.depth, cfg.channel
        self.project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
        self.train_file = self.project_path + '/data_preparation/CamVid/train.h5'
        self.valid_file = self.project_path + '/data_preparation/CamVid/valid.h5'
        self.test_file = self.project_path + '/data_preparation/CamVid/valid.h5'
        self.num_train = self.count_num_samples(mode='train')   # list of number of samples in each train file

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            img_idx = np.sort(np.random.choice(self.num_train, size=self.batch_size, replace=False))
            h5f = h5py.File(self.train_file, 'r')
            x = h5f['x'][list(img_idx)]
            y = h5f['y'][list(img_idx)]
            h5f.close()
            if self.augment:
                x, y = augmentation(x, y)
        elif mode == 'valid':
            h5f = h5py.File(self.valid_file, 'r')
            x = h5f['x'][start:end]
            y = h5f['y'][start:end]
            h5f.close()
        else:
            h5f = h5py.File(self.test_file, 'r')
            x = h5f['x'][start:end]
            y = h5f['y'][start:end]
            h5f.close()
        return x, y

    def count_num_samples(self, mode='valid'):
        if mode == 'train':     # count and store the number of samples from each train file
            print('counting the number of train samples........')
            h5f = h5py.File(self.train_file, 'r')
            num_ = h5f['y'][:].shape[0]
            h5f.close()
        elif mode == 'valid':
            h5f = h5py.File(self.valid_file, 'r')
            num_ = h5f['y'][:].shape[0]
        else:
            h5f = h5py.File(self.test_file, 'r')
            num_ = h5f['y'][:].shape[0]
        return num_


def augmentation(img_batch, mask_batch):
    img_batch_aug, mask_batch_aug = img_batch, mask_batch
    for i in range(img_batch.shape[0]):
        axis_aug = np.random.randint(2)
        image, mask = img_batch[i], mask_batch[i]
        if axis_aug:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        img_batch_aug[i] = image
        mask_batch_aug[i] = mask
    return img_batch_aug, mask_batch