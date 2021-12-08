import os
import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np

mask_dir = '../../GITCODE/yolact/output_coco'
coco_dir = '../../GITCODE/yolact/test2017'
au_dir = '../../datasets/casia2groundtruth/CASIA2.0_revised/Au'
# tp_dir = '../datasets/casia2groundtruth/CASIA2.0_revised/Tp'

class Augmentation(object) :
    def __init__(self):
        pass
    def __call__(self):
        pass

import sys
import torch
import torch.fft as fft

import cv2
import kornia.color as k_color
import numpy as np
import matplotlib.pyplot as plt

from time import time



def get_casia_dir(au_dir, prefix):
    files = os.listdir(au_dir)
    mask_list = []
    for file in files:
        au = os.path.join(au_dir, file)
        mask_list.append(au)
    # for file in files:
    #     tp = os.path.join(tp_dir, file)
    #     mask_list.append(tp)

    return mask_list

def get_mask_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    mask_list = []
    for file in files:
        path = os.path.join(root_dir, file)
        mask_list.append(path)
        if os.path.isdir(path):
            get_mask_dir(path, prefix + "    ")
    return mask_list

def get_img_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    img_list = []
    for file in files:
        path = os.path.join(root_dir, file)
        jpg = path.split('/')[-1].split('.')[0]
        img_path = os.path.join(coco_dir,jpg+'.jpg')
        img_list.append(img_path)
        if os.path.isdir(path):
            get_mask_dir(path, prefix + "    ")
    return img_list

img = get_img_dir(mask_dir, "")
mask = get_mask_dir(mask_dir, "")
casia = get_casia_dir(au_dir, "")
h, w = 256, 256

for i in range(len(casia)):
    source = cv2.imread(casia[i])
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    mask_read = cv2.imread(mask[i], 0)
    img_read = cv2.imread(img[i])
    img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)

    mask_read[mask_read<254] = 1
    mask_read[mask_read==255] = 0
    expand_mask = np.expand_dims(mask_read, axis=2)

    repeat = np.repeat(expand_mask, 3, axis=2)


    masked = img_read*repeat
    masked = cv2.resize(masked, dsize=(w, h), interpolation=cv2.INTER_AREA)
    mask_read = cv2.resize(mask_read, dsize=(w, h), interpolation=cv2.INTER_AREA)
    repeat = cv2.resize(repeat, dsize=(w, h), interpolation=cv2.INTER_AREA)
    source = cv2.resize(source, dsize=(w, h), interpolation=cv2.INTER_AREA)

    rows, cols = repeat.shape[:2]
    randint = np.random.randint(1,360)
    randf = float(np.round(np.random.rand(1), 1))
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), randint, randf * 0.9)
    repeat = cv2.warpAffine(repeat, M, (cols, rows))
    masked = cv2.warpAffine(masked, M, (cols, rows))

    x = np.random.randint(-150,150)
    y = np.random.randint(-150,150)
    T = np.float64([[1,0,x], [0,1,y]])
    repeat = cv2.warpAffine(repeat, T, (cols,rows))
    masked = cv2.warpAffine(masked, T, (cols,rows))

    if np.random.rand(1) > 0.5 :
        masked = cv2.GaussianBlur(masked, (3, 3), 0)

    repeat[repeat==1] = 3
    repeat[repeat==0] = 1
    repeat[repeat==3] = 0

    byn = source*repeat
    out_img = byn + masked
    out_mask = repeat + 1
    out_mask[out_mask==1]=255
    out_mask[out_mask==2]=0

    out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(out_img)
    # ax[1].imshow(out_mask, cmap='gray')
    # ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0]#.set_title('pred')
    # ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1]#.set_title('pred')
    # plt.tight_layout()
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    #
    # plt.show()

    # cv2.imwrite(f'../../datasets/coco_casia/splice/img{i}.png', out_img)
    # cv2.imwrite(f'../../datasets/coco_casia/gt/gt{i}.png', out_mask)