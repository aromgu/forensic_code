import random

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

class AuTp(Dataset) :
    def __init__(self, tp_path, au_path, tp_label_path, au_label_path, img_size, transform=None):
        super(AuTp, self).__init__()

        self.tp_path = tp_path
        self.au_path = au_path
        self.tp_label_path = tp_label_path
        self.au_label_path = au_label_path
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.tp_path)

    def __getitem__(self, idx):
        tp_image = np.array(Image.open(self.tp_path[idx]).convert('RGB'), dtype=np.uint8)
        au_image = np.array(Image.open(self.au_path[idx]).convert('RGB'), dtype=np.uint8)
        tp_label = np.array(Image.open(self.tp_label_path[idx]).convert('L'), dtype=np.uint8)
        if self.au_label_path[idx] == 'Au' :
            au_label = np.zeros((au_image.shape[0], au_image.shape[1], 1), dtype=np.uint8)
        # else :
        #     label = np.array(Image.open(self.label_path[idx]).convert('L'), dtype=np.uint8)

        if self.transform :
            seed = random.randint(0, 2**32)
            self._set_seed(seed); tp_image = self.transform(tp_image)
            self._set_seed(seed); au_image = self.transform(au_image)
            self._set_seed(seed); tp_label = self.transform(tp_label)
            self._set_seed(seed); au_label = self.transform(au_label)

        return (tp_image, au_image), (tp_label, au_label)
        # if self.label_path[idx] == 'Au':
        #     labelname = 'None'
        # prob = np.random.rand(1)
        # if prob > 0.5:
        #     return (tp_image, au_image), (tp_label, au_label)
        # else:
        #     return (au_image, tp_image), (au_label, tp_label)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)