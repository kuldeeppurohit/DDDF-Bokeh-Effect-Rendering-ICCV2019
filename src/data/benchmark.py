import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        list_lr2 = [[] for _ in self.scale]
        list_lr3 = [[] for _ in self.scale]
## kp editing
##        for entry in os.scandir(self.dir_hr):
        for entry in os.listdir(self.dir_hr):
## kp editing
##            filename = os.path.splitext(entry.name)[0]
            filename = os.path.splitext(entry)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(self.dir_lr,'{}{}'.format(filename, self.ext)  ))
                list_lr2[si].append(os.path.join(self.dir_lr2,'{}{}'.format(filename, self.ext)  ))
                list_lr3[si].append(os.path.join(self.dir_lr3,'{}{}'.format(filename, self.ext)  ))                 
                    ##'X{}/{}x{}{}'.format(s, filename, s, self.ext)            ))

        list_hr.sort()
        for l in list_lr:
            l.sort()
        for l in list_lr2:
            l.sort()
        for l in list_lr3:
            l.sort()            


        return list_hr, list_lr, list_lr2, list_lr3

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.dir_lr2 = os.path.join(self.apath, 'LR2')
        self.dir_lr3 = os.path.join(self.apath, 'LR3')        
        self.ext = '.jpg'
