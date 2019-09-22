import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np
import tables


class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', for_train=False,transforms='', return_target=True):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                # path = os.path.join(root, line , name + '_')
                path = os.path.join(root, line, '')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.return_target = return_target

        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]

        x, y = pkload(path + 'data_f32.pkl')
        # print(x.shape, y.shape)#(240, 240, 155, 4) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
        done = False
        if self.return_target:
            while not done:
                # print(x.shape, y.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
                a, b = self.transforms([x, y])
                # print(a.shape,b.shape)#(1, 128, 128, 128, 4) (1, 128, 128, 128)
                if b.sum() > 0:
                    done = True
                    x, y = a, b

        else:
            x = self.transforms(x)

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(x.shape, y.shape)  # (240, 240, 155, 4) (240, 240, 155)
        return x, y

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

class BraTSDataset_h5file(Dataset):
    def __init__(self, root, data_file, train_idx_file, val_idx_file):
        self.data_file_opened = tables.open_file(os.path.join(root, data_file), "r")
        self.train_idxs = pkload(os.path.join(root, train_idx_file))
        self.val_idxs = pkload(os.path.join(root, val_idx_file))

    def __getitem__(self, index):
        data = self.data_file_opened.root.data[index]
        truth = self.data_file_opened.root.truth[index, 0]
        data, truth = data[None, ...], truth[None, ...]

        x = torch.from_numpy(data)
        y = torch.from_numpy(truth)
        return x, y

    def __len__(self):
        return len(self.data_file_opened.root.data)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]