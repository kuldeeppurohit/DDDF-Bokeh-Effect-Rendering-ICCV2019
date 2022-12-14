import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class dvd(srdata.SRData):
    def __init__(self, args, train=True):
        super(dvd, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        list_lr2 = [[] for _ in self.scale]
        list_lr3 = [[] for _ in self.scale]
        '''
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val
        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
        '''
        for entry in os.listdir(self.dir_hr):
            filename = os.path.splitext(entry)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(self.dir_lr,'{}{}'.format(filename, self.ext)  ))
                list_lr2[si].append(os.path.join(self.dir_lr2,'{}{}'.format(filename, self.ext)  ))
                list_lr3[si].append(os.path.join(self.dir_lr3,'{}{}'.format(filename, self.ext)  ))                
##                    'X{}/{}{}'.format(s, filename, self.ext)  ))



        return list_hr, list_lr, list_lr2, list_lr3

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/dvd'
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.dir_lr2 = os.path.join(self.apath, 'LR2')
        self.dir_lr3 = os.path.join(self.apath, 'LR3')
        self.ext = '.jpg'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

