import os
import sys

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.Dataset import CASIADataset
from datasets.AuTp_Dataset import AuTp
# from datasets.patch_Dataset import PATCHDataset
from sklearn.model_selection import train_test_split
from utils.get_functions import *


args = get_init()

def load_dataloader(data_path, split_ratio, batch_size, img_size, num_workers) :
    # Tp_image_path = os.path.join(data_path, 'CASIA2.0_revised/Tp')
    # Tp_label_path = os.path.join(data_path, 'CASIA2.0_Groundtruth')
    Tp_image_path = os.path.join(data_path, 'casia2groundtruth/CASIA 2.0/Tp')
    Tp_label_path = os.path.join(data_path, 'casia2groundtruth/CASIA2.0_Groundtruth')
    # Au_image_path = os.path.join(data_path, 'CASIA2.0_revised/Au')
    Au_image_path = os.path.join(data_path, 'casia2groundtruth/CASIA 2.0/Au')

    Tp_images = sorted([Tp_image_path + '/' + x for x in os.listdir(Tp_image_path)]) #[:10]
    Tp_labels = sorted([Tp_label_path + '/' + x for x in os.listdir(Tp_label_path)]) #[:10]
    Au_images = sorted([Au_image_path + '/' + x for x in os.listdir(Au_image_path)]) #[:10]
    Au_labels = ['Au' for _ in range(len(Au_images))]

    Total_images = Tp_images + Au_images
    Total_labels = Tp_labels + Au_labels

    train_x, test_x, train_y, test_y = train_test_split(Total_images, Total_labels, test_size=split_ratio, shuffle=True)

    cnt = 0
    for label_ in train_y :
        if label_ == 'Au' :
            cnt += 1
    print("#train Tp = ", len(train_y) - cnt)
    print("#train Au = ", cnt)
    cnt = 0
    for label_ in test_y :
        if label_ == 'Au' :
            cnt += 1
    print("#test Tp = ", len(test_y) - cnt)
    print("#test Au = ", cnt)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10, expand=False),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
# AuTp
    train_dataset = CASIADataset(train_x, train_y, img_size, train_transform)
    test_dataset = CASIADataset(test_x, test_y, img_size, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print('create data loader success')

    return train_loader, test_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_autp(data_path, split_ratio, batch_size, img_size, num_workers) :
    # Tp_image_path = os.path.join(data_path, 'casia2groundtruth/CASIA2.0_revised/Tp')
    # Tp_label_path = os.path.join(data_path, 'casia2groundtruth/CASIA2.0_Groundtruth')
    # Au_image_path = os.path.join(data_path, 'casia2groundtruth/CASIA2.0_revised/Au')
    Tp_image_path = os.path.join(data_path, 'casia2groundtruth/CASIA 2.0/Tp')
    Tp_label_path = os.path.join(data_path, 'casia2groundtruth/CASIA2.0_Groundtruth')
    Tp_images = sorted([Tp_image_path + '/' + x for x in os.listdir(Tp_image_path)]) #[:50]
    Tp_labels = sorted([Tp_label_path + '/' + x for x in os.listdir(Tp_label_path)]) #[:50]

    tp_list = []
    for i in Tp_images:
        tp_name1 = i.split('/')[-1].split('.')[0].split('_')[5]
        tp_name = '_'.join([tp_name1[:3], tp_name1[3:]])
        tp_list.append(tp_name)

    Au_image_path = os.path.join(data_path, 'casia2groundtruth/CASIA 2.0/Au')

    au_name = []
    for i in tp_list:
        for j in os.listdir(Au_image_path):
            if i in j:
                au_name.append(j)

    Au_images = [Au_image_path + '/' + x for x in au_name] #[:50]

    Au_labels = ['Au' for _ in range(len(Au_images))]

    # Total_images = Tp_images + Au_images
    # Total_labels = Tp_labels + Au_labels

    # train_x, test_x, train_y, test_y = train_test_split(Total_images, Total_labels, test_size=split_ratio, shuffle=False)

    tp_train_x, tp_test_x = data_split(Tp_images)
    au_train_x, au_test_x = data_split(Au_images)
    tp_train_y, tp_test_y = data_split(Tp_labels)
    au_train_y, au_test_y = data_split(Au_labels)

  #  if args.train :
     # train_x = train_x# + coco_images
     # train_y = train_y# + coco_labels
    cnt = 0
    for label_ in au_train_x :
        if label_ == 'Au' :
            cnt += 1
    print("#train Tp = ", len(tp_train_x) - cnt)
    print("#train Au = ", cnt)
    cnt = 0
    for label_ in au_test_y :
        if label_ == 'Au' :
            cnt += 1
    print("#test Tp = ", len(tp_test_x) - cnt)
    print("#test Au = ", cnt)
    if args.aug_option == 'y':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            Augmentation(),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = AuTp(tp_train_x, au_train_x, tp_train_y, au_train_y, img_size, train_transform)
    test_dataset = CASIADataset(tp_test_x + au_test_x, tp_test_y + au_test_y, img_size, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)

    print('create data loader success')

    return train_loader, test_loader