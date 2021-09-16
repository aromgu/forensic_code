# 1. [무료배송] 에어팟프로 애플로고 실리콘 케이스
# 2. 도낫도낫 와일드 에어팟 프로 실리콘 케이스
# 3. 누아트 페블 에어팟프로 실리콘케이스 + 메탈 철가루 방지스티커 랜덤 발송

import random
import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

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
    parser = argparse.ArgumentParser()

    # ETC
    parser.add_argument('--data_path', default='../datasets/casia2groundtruth')
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--parent_dir', type=str, default='./saved_models/', help='for saving trained models')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_root', type=str, default='patch32')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--saved_pt', type=str, default='last_100.pth')

    # Train Parser Args
    parser.add_argument('--model_name', default='RRU', help='model architecture name')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--criterion', type=str, default='FL')

    # Mask Generation Process hyperparameters
    parser.add_argument('--angle', type=int, default=30, help='parameter for MGP module')
    parser.add_argument('--length', type=int, default=10, help='parameter for MGP module')
    parser.add_argument('--preserve_range', type=int, default=50, help='parameter for MGP module')
    parser.add_argument('--num_enc', type=int, default=2, help='parameter for MGP module')
    parser.add_argument('--beta', type=float, default=0.00001, help='Cutout parameter')

    parser.add_argument('--fad_option', default='n', help='FAD option')
    parser.add_argument('--mgp_option', default='n', help='MGP option')
    parser.add_argument('--patch_option', type=str, default='y')

    parser.add_argument('--img_size', default=256, type=int, help='image width and height size')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--erase_prob', type=float, default=0.5)

    parser.add_argument('--high_loss_weight', type=float, default=0.3)
    parser.add_argument('--low_loss_weight', type=float, default=0.7)

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
    else :
        print("You choose wrong optimizer : [BCE, FL]")
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