import random
import argparse

import torch
import torch.nn as nn
from torch import optim

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
    parser.add_argument('--data_path', default='/media/jhnam19960514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection/CASIA/casia2groundtruth')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu')))
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--parent_dir', type=str, default='./saved_models/', help='for saving trained models')

    # Train Parser Args
    parser.add_argument('--model_name', default='CNN', help='model architecture name')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--angle', type=int, default=30, help='parameter for MGP module')
    parser.add_argument('--length', type=int, default=10, help='parameter for MGP module')
    parser.add_argument('--preserve_range', type=int, default=0, help='parameter for MGP module')
    parser.add_argument('--num_enc', type=int, default=5, help='parameter for MGP module')


    parser.add_argument('--patch_option', default='n', help='illumination option')
    parser.add_argument('--fad_option', default='n', help='FAD option')
    parser.add_argument('--mgp_option', default='y', help='MGP option')

    parser.add_argument('--img_size', default=256, type=int, help='image width and height size')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--net_loss_weight', type=float, default=0.5)
    parser.add_argument('--low_loss_weight', type=float, default=0.5)

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        nn.init.constant_(m.bias.data, 0)

def get_current_lr(optimizer) :
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr(step, total_step, lr_max, lr_min) :
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * ( 1 + np.cos(step/total_step * np.pi))

def get_criterion(args) :
    criterion = nn.BCEWithLogitsLoss().to(args.device)

    return criterion

def get_optimizer(args, model) :
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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