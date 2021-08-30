# 1. [무료배송] 에어팟프로 애플로고 실리콘 케이스
# 2. 도낫도낫 와일드 에어팟 프로 실리콘 케이스
# 3. 누아트 페블 에어팟프로 실리콘케이스 + 메탈 철가루 방지스티커 랜덤 발송

import os
import random
import argparse

import torch
import torch.nn as nn

import numpy as np

# 1 Epoch 당 테스트 이미지 인퍼런스
# 서브 플롯으로 1 x 5 짜리 5장 인퍼런스 하는 거지~^^*;;""
# 학습할 때 random crop? transforms.RandomCrop()

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

def get_init():
    root_dataset_path = '/media/jhnam19960514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection/CASIA/casia2groundtruth'
    parser = argparse.ArgumentParser()

    # ETC
    parser.add_argument('--Au_image_path', default=os.path.join(root_dataset_path, 'CASIA2.0_revised/Au'))
    parser.add_argument('--Tp_image_path', default=os.path.join(root_dataset_path, 'CASIA2.0_revised/Tp'))
    parser.add_argument('--Tp_label_path', default=os.path.join(root_dataset_path, 'CASIA2.0_Groundtruth'))
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu')))
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--parent_dir', type=str, default='./saved_models/', help='for saving trained models')

    # Train Parser Args
    parser.add_argument('--model_name', default='Unet', help='model architecture name')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--patch_option', default='n', help='illumination option')
    parser.add_argument('--fad_option', default='n', help='FAD option')
    parser.add_argument('--img_size', default=256, type=int, help='image width and height size')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    parser.add_argument('--net_loss_weight', type=float, default=1.)
    parser.add_argument('--patch_loss_weight', type=float, default=0.)

    args = parser.parse_args()
    return args

def fix_seed(seed, device) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Your experiment is fixed to {seed}")

def get_current_lr(optimizer) :
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr(step, total_step, lr_max, lr_min) :
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * ( 1 + np.cos(step/total_step * np.pi))

def get_scheduler(args, train_loader, optimizer) :
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    return scheduler