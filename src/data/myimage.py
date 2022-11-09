import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        apath1 = args.testpath + '/' + args.testset + '/LR'#/X' + str(args.scale[0])
        apath2 = args.testpath + '/' + args.testset + '/LR2'#/X' + str(args.scale[0])
        apath3 = args.testpath + '/' + args.testset + '/LR3'#/X' + str(args.scale[0])        
        self.filelist1 = []
        self.filelist2 = []
        self.filelist3 = []     
        self.imnamelist = []
        if not train:
            for f in os.listdir(apath1):
                try:
                    filename = os.path.join(apath1, f)
                    misc.imread(filename)
                    self.filelist1.append(filename)
                    filename = os.path.join(apath2, f)
                    misc.imread(filename)
                    self.filelist2.append(filename)
                    filename = os.path.join(apath3, f)
                    misc.imread(filename)
                    self.filelist3.append(filename)
                    self.imnamelist.append(f)
                except:
                    pass

    def __getitem__(self, idx):

        filename = os.path.split(self.filelist1[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.filelist1[idx])
        lr1 = common.set_channel([lr], self.args.n_colors)[0]

        filename = os.path.split(self.filelist2[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.filelist2[idx])
        lr2 = common.set_channel([lr], self.args.n_colors)[0] 

        filename = os.path.split(self.filelist3[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.filelist3[idx])
        lr3 = common.set_channel([lr], self.args.n_colors)[0] 

        return common.np2Tensor([lr1], self.args.rgb_range)[0],common.np2Tensor([lr2], self.args.rgb_range)[0],common.np2Tensor([lr3], self.args.rgb_range)[0], -1, filename
    def __len__(self):
        #print('hi',len(self.filelist))
        return len(self.filelist1)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

