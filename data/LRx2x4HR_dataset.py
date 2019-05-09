import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from collections import OrderedDict
import h5py
import logging

class LRx2x4HRDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRx2x4HRDataset, self).__init__()
        self.logger = logging.getLogger('base')
        self.opt = opt

        self.HR_hdf5 = OrderedDict()
        self.LRx2_hdf5 = OrderedDict()
        self.LRx4_hdf5 = OrderedDict()

        # read SM from hdf5 file
        with h5py.File(opt['dataroot_HR']) as hf:
            self.logger.info('Read hdf5: {}'.format(opt['dataroot_HR']))
            for key, value in hf.items():
                self.HR_hdf5[key] = np.array(value)
        with h5py.File(opt['dataroot_LRx2']) as hf:
            self.logger.info('Read hdf5: {}'.format(opt['dataroot_LRx2']))
            for key, value in hf.items():
                self.LRx2_hdf5[key] = np.array(value)
        with h5py.File(opt['dataroot_LRx4']) as hf:
            self.logger.info('Read hdf5: {}'.format(opt['dataroot_LRx4']))
            for key, value in hf.items():
                self.LRx4_hdf5[key] = np.array(value)

        assert len(self.HR_hdf5) > 0, 'Error: HR is empty.'
        assert len(self.LRx2_hdf5) > 0, 'Error: LRx2 is empty.'
        assert len(self.LRx4_hdf5) > 0, 'Error: LRx4 is empty.'

        if self.opt['phase'] == 'val':
            self.HR_hdf5['data'] = np.array(self.HR_hdf5['data'])
            self.LRx2_hdf5['data'] = np.array(self.LRx2_hdf5['data'])
            self.LRx4_hdf5['data'] = np.array(self.LRx4_hdf5['data']) - 236.17393 #input norm
        else:
            self.HR_hdf5['data'] = np.array(self.HR_hdf5['data'])
            self.LRx2_hdf5['data'] = np.array(self.LRx2_hdf5['data'])
            self.LRx4_hdf5['data'] = np.array(self.LRx4_hdf5['data']) - 236.17393 #input norm
        if 'data' in self.HR_hdf5 and 'data' in self.LRx2_hdf5 and 'data' in self.LRx4_hdf5:
            assert len(self.HR_hdf5['data']) == len(self.LRx2_hdf5['data']) == len(self.LRx4_hdf5['data']), \
                'HR and LRx2 and LRx4 datasets have different number of images - {}, {}, {}.'.format(
                len(self.HR_hdf5['data']), len(self.LRx2_hdf5['data']), len(self.LRx4_hdf5['data']))


        self.random_scale_list = [1]

    def __getitem__(self, index):
        # get HR image
        # load frequence as BGR, HWDC
        img_HR = self.HR_hdf5['data'][index]

        # get LR image
        # load frequence as BGR, HWDC
        if 'data' in self.LRx2_hdf5:
            img_LRx2 = self.LRx2_hdf5['data'][index]
        if 'data' in self.LRx4_hdf5:
            img_LRx4 = self.LRx4_hdf5['data'][index]

        if self.opt['phase'] == 'train':
            # augmentation - flip, rotate
            img_LRx4, img_LRx2, img_HR = util.augment(
                [img_LRx4, img_LRx2, img_HR], self.opt['use_flip'], self.opt['use_rot']
            )

        # BGR to RGB,
        if img_HR.shape[3] == 3:
            img_HR = img_HR[:, :, :, [2, 1, 0]]
            img_LRx2 = img_LRx2[:, :, :, [2, 1, 0]]
            img_LRx4 = img_LRx4[:, :, :, [2, 1, 0]]
        # HWC to CHW, numpy to tensor
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (3, 0, 1, 2)))).float()
        img_LRx2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LRx2, (3, 0, 1, 2)))).float()
        img_LRx4 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LRx4, (3, 0, 1, 2)))).float()

        return {'LRx4': img_LRx4, 'LRx2': img_LRx2, 'HR': img_HR}

    def __len__(self):
        return len(self.HR_hdf5['data'])
