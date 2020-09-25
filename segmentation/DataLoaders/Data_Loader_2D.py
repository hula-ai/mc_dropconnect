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
        self.train_file = self.project_path + '/data_preparation/our_data/6_2d/train_2d'
        self.valid_file = self.project_path + '/data_preparation/our_data/6_2d/test_2d.h5'
        self.test_file = self.project_path + '/data_preparation/our_data/6_2d/test_2d.h5'
        # self.num_train = self.count_num_samples(mode='train')   # list of number of samples in each train file
        self.num_train = [8130, 8967, 7780, 8197, 8252, 7499]   # list of number of samples in each train file

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            train_num = np.random.randint(len(self.num_train))
            img_idx = np.sort(np.random.choice(self.num_train[train_num], size=self.batch_size, replace=False))
            h5f = h5py.File(self.train_file + '_' + str(train_num) + '.h5', 'r')
            x = h5f['x_norm'][list(img_idx)]
            y = h5f['y'][list(img_idx)]
            h5f.close()
        elif mode == 'valid':
            h5f = h5py.File(self.valid_file, 'r')
            x = h5f['x_norm'][start:end]
            y = h5f['y'][start:end]
            h5f.close()
        else:
            h5f = h5py.File(self.test_file, 'r')
            x = h5f['x_norm'][start:end]
            y = h5f['y'][start:end]
            h5f.close()
        return x, y

    def count_num_samples(self, mode='valid'):
        if mode == 'train':     # count and store the number of samples from each train file
            print('counting the number of train samples........')
            l = len(glob.glob(self.project_path + '/data_preparation/our_data/6_2d/*.h5')) - 1  # number of train files
            num_ = []
            for i in range(l):
                h5f = h5py.File(self.train_file + '_' + str(i) + '.h5', 'r')
                num_.append(h5f['y'][:].shape[0])
                h5f.close()
        elif mode == 'valid':
            h5f = h5py.File(self.valid_file, 'r')
            num_ = h5f['y'][:].shape[0]
        else:
            h5f = h5py.File(self.test_file, 'r')
            num_ = h5f['y'][:].shape[0]
        return num_
