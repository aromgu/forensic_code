import random

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

class ilove_Dataset(Dataset) :
    def __init__(self, image_path, label_path, img_size, transform=None):
        super(ilove_Dataset, self).__init__()

        self.image_path = image_path
        self.label_path = label_path
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # imgname = self.image_path[idx]
        # labelname = self.label_path[idx]

        image = np.array(Image.open(self.image_path[idx]).convert('RGB'), dtype=np.uint8)
        if self.label_path[idx] == 'Au' :
            label = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        else :
            label = np.array(Image.open(self.label_path[idx]).convert('L'), dtype=np.uint8)

        if self.transform :
            seed = random.randint(0, 2**32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.transform(label)
        # if self.label_path[idx] == 'Au':
        #     labelname = 'None'
        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class DSO_Dataset(Dataset) :
    def __init__(self, image_path, label_path, img_size, transform=None):
        super(DSO_Dataset, self).__init__()

        self.image_path = image_path
        self.label_path = label_path
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # imgname = self.image_path[idx]
        # labelname = self.label_path[idx]

        image = np.array(Image.open(self.image_path[idx]).convert('RGB'), dtype=np.uint8)
        if self.label_path[idx] == 'Au' :
            label = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        else :
            label = 255 - np.array(Image.open(self.label_path[idx]).convert('L'), dtype=np.uint8)

        if self.transform :
            seed = random.randint(0, 2**32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.transform(label)
        # if self.label_path[idx] == 'Au':
        #     labelname = 'None'
        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)