import os
import cv2
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from torchvision import transforms

class CreateDataset(Dataset):
    def __init__(self, image_path, label_path, img_size):
        self.image_path = image_path
        # self.depth_path = []
        # for i in image_path:
        #     i = os.path.join('datasets/depth/',"/".join(i.split('/')[-2:]).split('.')[0]+".png")
        #     self.depth_path.append(i)
        self.label_path = label_path
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(256, 256),
            transforms.Resize((self.img_size,self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, expand=False),
            transforms.ToTensor()
        ])

        self.label_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(256, 256),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, expand=False),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # image.shape = [320, 384] = h, w
        image = np.array(Image.open(self.image_path[idx]).convert('RGB'), dtype=np.uint8)
        # depth = np.array(Image.open(self.depth_path[idx]).convert('RGB'), dtype=np.uint8)
        # raw = torch.LongTensor(cv2.resize(image, (384, 320)))
        if self.label_path[idx] == 'ssibal':
            label = np.zeros(shape=(self.img_size, self.img_size), dtype=np.uint8)
            # label = np.zeros(shape=(256,256), dtype=np.uint8)
            classify_y = torch.zeros(1)
        else:
            label = np.array(Image.open(self.label_path[idx]).convert('L'), dtype=np.uint8)
            classify_y = torch.ones(1)

        # depth = self.transform(depth)

        seed = random.randint(0, 2**32)
        self._set_seed(seed); X = self.transform(image)
        self._set_seed(seed); y = self.label_transform(label)

        # y[y > 0.25] = 1     # seg label resizing 시 threshold 지정
        y = y.float()    # label 의 채널 삭제
        return X, y

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)


def load_path(Au_image_path, Tp_image_path, Tp_label_path, split_ratio, batch_size, img_size):

    Au_images = sorted([Au_image_path + '/' + x for x in os.listdir(Au_image_path)])
    Au_labels = ['ssibal' for _ in range(len(Au_images))]
    Tp_images = sorted([Tp_image_path + '/' + x for x in os.listdir(Tp_image_path)])#[:10]
    Tp_labels = sorted([Tp_label_path + '/' + x for x in os.listdir(Tp_label_path)])#[:10]

    # total_images = Au_images + Tp_images
    # total_labels = Au_labels + Tp_labels
    total_images = Tp_images
    total_labels = Tp_labels
    train_x, test_x, train_y, test_y = train_test_split(total_images, total_labels, test_size=split_ratio, shuffle=True)

    # train_set = CreateDataset(Tp_images, Tp_labels)
    # test_set = CreateDataset(Tp_images, Tp_labels)
    #
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


    train_set = CreateDataset(train_x, train_y, img_size)
    test_set = CreateDataset(test_x, test_y, img_size)
    infer_set = CreateDataset(test_x, test_y, img_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    infer_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print('create data loader success')

    return train_loader, test_loader, infer_loader


if __name__ == '__main__':
    train_loader, test_loader, infer_loader = load_path(Au_image_path='../datasets/casia2groundtruth/CASIA2.0_revised/Au',
              # Au_depth_path='../datasets/depth/Au',
              Tp_image_path='../datasets/casia2groundtruth/CASIA2.0_revised/Tp',
              # Tp_depth_path='../datasets/depth/Tp',
              Tp_label_path='../datasets/casia2groundtruth/CASIA2.0_Groundtruth',
              split_ratio=0.2, batch_size=4)

    for X, y in train_loader:
        print(X.shape)
        print(y.shape)
        break