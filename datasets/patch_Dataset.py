import random

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2

class PATCHDataset(Dataset) :
    def __init__(self, image_path, label_path, img_size, patch_size, train=True, transform=None):
        super(PATCHDataset, self).__init__()

        self.image_path = image_path
        self.label_path = label_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.train = train
        self.transform = transform
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_path[idx]).convert('RGB'), dtype=np.uint8)
        if self.label_path[idx] == 'Au' :
            label = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        else :
            label = np.array(Image.open(self.label_path[idx]).convert('L'), dtype=np.uint8)

        # label = np.array(Image.open(self.label_path[idx]).convert('L'), dtype=np.uint8)

        if image.shape[0:2] != label.shape[0:2]:
            print(image.shape, label.shape)
            print('target and input size is different')
            print(self.image_path[idx])
            label = cv2.resize(label,(image.shape[0],image.shape[1]))

        if self.transform :
            seed = random.randint(0, 2**32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.transform(label)

        if self.train :
            image = self.extract_patch(image, 3)
            label = self.extract_patch(label, 1)

        return image, label


    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)