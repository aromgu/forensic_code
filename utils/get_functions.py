# 1. [무료배송] 에어팟프로 애플로고 실리콘 케이스
# 2. 도낫도낫 와일드 에어팟 프로 실리콘 케이스
# 3. 누아트 페블 에어팟프로 실리콘케이스 + 메탈 철가루 방지스티커 랜덤 발송

import random
import argparse
import sys
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch.nn.functional as F
from utils import *

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.BCEWithLogitsLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()
    def forward(self, pred, target):
        smooth = 1e-5
        # start_time = time()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = 2.0 * (intersection + smooth) / (union + smooth)

        dice_loss = 1.0 - dice.mean()
        loss = bce + dice_loss

        # end_time = time() - start_time
        #
        # print("time", end_time)

        return loss, bce, dice_loss

def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
     between two unordered sets of points (the function is symmetric).
     Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1.cpu().detach())
    set2 = np.array(set2.cpu().detach())

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res

def hausdorff(gt,pred):
    aver_edge_loss = 0.0
    for gt_, pred_ in zip(gt, pred):
        gt_ = gt_.squeeze()
        pred_ = pred_.squeeze()
        edge_loss = averaged_hausdorff_distance(gt_, pred_)
        aver_edge_loss += edge_loss
    aver_edge_loss /= gt.size(0)
    return edge_loss

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train segmentation network')
#
#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         required=True,
#                         type=str)
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument('opts',
#                         help="Modify config options using the command-line",
#                         default=None,
#                         nargs=argparse.REMAINDER)
#
#
#     args = parser.parse_args()
#     update_config(config, args)
#
#     return args


def get_init():
    parser = argparse.ArgumentParser()

    # ETC
    # parser.add_argument('--data_path', default='/home/sam/Desktop/RM/datasets')
    parser.add_argument('--data_path', default='../datasets')
    # parser.add_argument('--data_path', default='/media/semi/a3840fc5-2c0b-494d-9fed-f193ec93a37e/home/sam/Desktop/RM/datasets')
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--parent_dir', type=str, default='./saved_models/', help='for saving trained models')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_root', default='18ASPPfilter150')
    parser.add_argument('--train', action='store_true', default=False) # python3 main.py --train => 학습하는 거고 | python3 main.py => 테스트 하는 거임 ㅇㅋ 입력
    parser.add_argument('--saved_pt', type=str, default='last_100.pth')

    # Train Parser Args
    parser.add_argument('--model_name', default='res18rfam', help='model architecture name')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--criterion', type=str, default='DICE')
    parser.add_argument('--trainer', default='fuse', help='fuse, split, autp')
    parser.add_argument('--dataloader', default='fuse', help='fuse, autp')

    # parser.add_argument('--fad_option', default='n', help='FAD option')
    parser.add_argument('--aug_option', default='n', help='MGP option')
    # parser.add_argument('--erase_prob', type=float, default=0.)
    # parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--diagonal', type=int, default=100)

    parser.add_argument('--img_size', default=256, type=int, help='image width and height size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()
    return args

def fix_seed(device) :
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)
    if device == 'cuda':
        torch.cuda.manual_seed_all(4321)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Your experiment is fixed to {4321}")


def get_trainer(trainer):
    if trainer == 'autp':
        from Trainer_autp import fit, test_epoch
    elif trainer == 'fuse':
        from Trainer_fusion import fit, test_epoch
    elif trainer == 'split':
        from Trainer_splittest import fit, test_epoch
    return fit, test_epoch

def get_dataloader(args):
    if args.dataloader == 'autp':
        train_loader, test_loader = load_autp(data_path=args.data_path,
                                                split_ratio=args.split_ratio,
                                                batch_size=args.batch_size,
                                                img_size=args.img_size,
                                                num_workers=args.num_workers)
    elif args.dataloader == 'fuse':
        from utils.load_functions import load_dataloader
        train_loader, test_loader = load_dataloader(data_path=args.data_path,
                                                split_ratio=args.split_ratio,
                                                batch_size=args.batch_size,
                                                img_size=args.img_size,
                                                num_workers=args.num_workers)
    return train_loader, test_loader

def get_device() :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"your device is {device}")

    return device

def get_current_lr(optimizer) :
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr(step, total_step, lr_max, lr_min) :
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * ( 1 + np.cos(step/total_step * np.pi))

def get_criterion(args) :
    if args.criterion == 'BCE' :
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'FL' :
        criterion = FocalLoss()
    elif args.criterion == 'DICE':
        criterion = Dice_loss()
    else :
        print("You choose wrong criterion : [BCE, FL, DICE]")
        sys.exit()

    print("Your criterion is : ", criterion)

    return criterion

def get_optimizer(args, model) :
    if args.optimizer == 'Adam' :
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD' :
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)
    else :
        print("You choose wrong optimizer : [Adam, SGD]")
        sys.exit()

    print("Your optimizer is : ", optimizer)

    return optimizer

def get_scheduler(args, train_loader, optimizer) :
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    return scheduler

def get_history() :
    history = dict()
    history['train_loss'] = list()
    history['test_loss'] = list()

    return history

def get_patches(image, channel, patch_size):
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = unfold(image)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, channel, patch_size, patch_size) # [P*B, C, W, H]
    return patches

import os
import cv2



def get_mask_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    mask_list = []
    for file in files:
        path = os.path.join(root_dir, file)
        mask_list.append(path)
        if os.path.isdir(path):
            get_mask_dir(path, prefix + "    ")
    return mask_list

def get_coco_dir(root_dir, prefix):
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

coco_dir = '../datasets/test2017'
mask_dir = '../datasets/output_coco'

def data_split(split, ratio=0.2):
    # idx = np.arange(0, len(split))
    # np.random.shuffle(idx)
    length = int(len(split) * ratio)
    train_data = split[length:]
    test_data = split[:length]
    return train_data, test_data

class Augmentation(object) :
    def __init__(self):
        super(Augmentation, self).__init__()

        self.coco = get_coco_dir(mask_dir, "")
        self.mask = get_mask_dir(mask_dir, "")

    def __call__(self, img):
        if np.random.rand(1) >= 0.5:
            h, w = 256, 256
            index = np.random.randint(1,len(self.mask))
            mask_data = cv2.imread(self.mask[index], 0)
            coco_data = cv2.imread(self.coco[index])

            mask_data[mask_data < 254] = 1
            mask_data[mask_data == 255] = 0
            expand_mask = np.expand_dims(mask_data, axis=2)
            mask_data = np.repeat(expand_mask, 3, axis=2)

            # masked = cv2.resize(masked, dsize=(w, h), interpolation=cv2.INTER_AREA)
            coco_data = cv2.resize(coco_data, dsize=(w, h), interpolation=cv2.INTER_AREA)
            mask_data = cv2.resize(mask_data, dsize=(w, h), interpolation=cv2.INTER_AREA)

            masked = coco_data * mask_data

            rows, cols = mask_data.shape[:2]
            randint = np.random.randint(1, 360)
            # randf = float(np.round(0.5 * np.random.rand(1), 1)) # [0.5 ~ 2.0] # scale
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), randint, 1)

            mask_data = cv2.warpAffine(mask_data, M, (cols, rows))
            masked = cv2.warpAffine(masked, M, (cols, rows))

            x = np.random.randint(-150, 150)
            y = np.random.randint(-150, 150)

            T = np.float64([[1, 0, x], [0, 1, y]])
            mask_data = cv2.warpAffine(mask_data, T, (cols, rows))
            masked = cv2.warpAffine(masked, T, (cols, rows))

            if np.random.rand(1) > 0.5:
                masked = cv2.GaussianBlur(masked, (3, 3), 0)

            mask_data[mask_data == 1] = 3
            mask_data[mask_data == 0] = 1
            mask_data[mask_data == 3] = 0
            img = np.array(img)

            if len(img.shape) == 2:
                out_mask = mask_data + 1
                out_mask[out_mask == 1] = 1
                out_mask[out_mask == 2] = 0

                img[img > 0] = 1
                mask = np.bitwise_or(img, out_mask[:,:,1])
                return mask
            else :
                masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
                blank = img*mask_data
                out_img = blank + masked
                # plt.imshow(out_img)
                # plt.show()
                return out_img
        else:
            return img