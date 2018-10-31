import numpy as np
import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
from torch.autograd import Variable

from tools import *

class Dataset(data.Dataset):
    def __init__(self, transfP, train):
        self.transfP = transfP
        self.files = {}
        self.labels = {}
        self.idx = 0
        path = GHOST_TRAIN if train else GHOST_TEST
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file and file[-3:] == 'png':
                    self.files[self.idx] = path + '/' + file
                    self.labels[self.idx] = 1
                    self.idx += 1
        path = NGHOST_TRAIN if train else NGHOST_TEST
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file and file[-3:] == 'png':
                    self.files[self.idx] = path + '/' + file
                    self.labels[self.idx] = 0
                    self.idx += 1
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        res = np.array([self.transfP(get_img(self.files[index]))])
        res = torch.from_numpy(res)
        res = res.type('torch.FloatTensor')
        target = torch.from_numpy(np.array([self.labels[index]]))
        target = target.type('torch.FloatTensor')
        return res, target, Variable(torch.Tensor([index]))
