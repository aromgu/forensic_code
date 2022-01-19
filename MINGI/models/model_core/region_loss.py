import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# CUDA_LAUNCH_BLOCKING=1

class Region_loss(nn.Module):
    def __init__(self):
        super(Region_loss, self).__init__()

    def forward(self, pred, targets):
        b,c,w,h = pred.shape

        loss_ = 0.0
        for i in range(b):
            pred_ = torch.flatten(pred, start_dim=1)[i]
            targets_ = torch.flatten(targets, start_dim=1)[i]
            pred_ = F.sigmoid(pred_)
            pred_[pred_ >= 0.5] = 1
            pred_[pred_ < 0.5] = 0

            index = (targets_ == 1).nonzero(as_tuple=True)[0]
            if index is None :
                loss = 0
                loss_ += loss
            else:
                nonzero = torch.count_nonzero(pred_[index])
                loss = 1 - (nonzero + 1e-6)/(len(index) + 1e-6)
                loss = Variable(loss, requires_grad=True)
                loss_ += loss

        return torch.abs(loss_) / b

if __name__ == '__main__':
    model = Region_loss().cuda()
    target = torch.ones((2, 1, 15, 15)).cuda() # 내말이
    pred = torch.zeros((2, 1, 15, 15)).cuda() # 내말이
    out = model(pred, target)
