import os

import cv2
from PIL import Image
from utils import *
# import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from load_functions import load_cococasia

args = get_init()
train_loader, test_loader = load_cococasia(data_path=args.data_path,
                                           split_ratio=args.split_ratio,
                                           batch_size=args.batch_size,
                                           img_size=args.img_size,
                                           num_workers=args.num_workers)


# JPG SAVE ===================
root = '/home/sam/Desktop/RM/datasets/casia2groundtruth/CASIA 2.0/Tp'

for idx, (img, label, name) in enumerate(train_loader):
    # print(name[0][0].split('/')[-1].split('.')[0])
    name = name[0][0].split('/')[-1].split('.')[0]
    # img = np.array(img)
    # img = transforms.ToPILImage(img)
    img = F.to_pil_image(img.squeeze())
    img = img.resize((256, 256))

    if 'Tp' in name:
        file = '/home/sam/Desktop/RM/datasets/casia2groundtruth/CASIA 2.0/Tp_jpg/{name}.jpg'.format(name=name)
        img.save(file, quality=100)

for idx, (img, label, name) in enumerate(test_loader):
    # print(name[0][0].split('/')[-1].split('.')[0])
    name = name[0][0].split('/')[-1].split('.')[0]
    # img = np.array(img)
    # img = transforms.ToPILImage(img)
    img = F.to_pil_image(img.squeeze())
    img = img.resize((256, 256))

    if 'Tp' in name:
        file = '/home/sam/Desktop/RM/datasets/casia2groundtruth/CASIA 2.0/Tp_jpg/{name}.jpg'.format(name=name)
        img.save(file, quality=100)

# TRAIN LIST TXT
# Au_train_txt = open('c2_au_train.txt', 'w')
# # Tp_train_txt = open('c2_tp_train.txt', 'w')
# # f = open('v2_auth_train_list.txt', 'w')
# root = 'CASIA 2.0'
#
# for idx, (image, label, name) in enumerate(train_loader): # image, label
#     train_img = str(name[0][0].split('/')[-1])
#     if 'Au' in train_img:
#         if train_img.split('.')[-1] == 'bmp':
#             continue
#         temp = ','.join(['CASIA 2.0/Au/{}'.format(train_img), str(None), 'CASIA 2.0/Au/{}'.format(train_img)])
#         Au_train_txt.write(temp+'\n')
#
#     if 'Tp' in train_img:
#         dummy = train_img.split('/')
#         image_name = dummy[-1].split('.')[0]
#         gt_name = f'CASIA 2 Groundtruth/{image_name}_gt.png'
#         jpg_name = f'{root}/Tp_jpg/{image_name}.jpg'
#         temp = ','.join([train_img, gt_name, jpg_name])
#         Tp_train_txt.write(temp + '\n')

# TEST LIST TXT ===============================

# Au_test_txt = open('c2_au_val.txt', 'w')
#
# for idx, (image, label, name) in enumerate(test_loader): # image, label
#     test_img = str(name[0][0].split('/')[-1])
#     if 'Au' in test_img:
#         if test_img.split('.')[-1] == 'bmp':
#             continue
#         temp = ','.join(['CASIA 2.0/Au/{}'.format(test_img), str(None), 'CASIA 2.0/Au/{}'.format(test_img)])
#         print(temp)
#         Au_test_txt.write(temp+'\n')

# Tp_test_txt = open('c2_tp_test.txt', 'w')
#     if 'Tp' in test_img:
#         dummy = test_img.split('/')
#         image_name = dummy[-1].split('.')[0]
#         gt_name = f'CASIA2.0_Groundtruth/{image_name}_gt.png'
#         jpg_name = f'{root}/Tp_jpg/{image_name}.jpg'
#         temp = ','.join([test_img, gt_name, jpg_name])
#         Tp_test_txt.write(temp + '\n')
#
print('done')