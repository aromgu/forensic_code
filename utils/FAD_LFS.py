import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Filter Module =======
class LFilter(nn.Module):
    def __init__(self, size, diagonal, use_learnable=True, norm=False):
        super(LFilter, self).__init__()
        self.use_learnable = use_learnable

        # low = generate_filter(band_start, band_end, size)
        low = create_filter(img_size=256, diagonal=diagonal)
        self.base = nn.Parameter(low, requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(low), requires_grad=False)
    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt

        return y

class HFilter(nn.Module):
    def __init__(self, size, diagonal, use_learnable=True, norm=False):
        super(HFilter, self).__init__()
        self.use_learnable = use_learnable

        high = create_filter(img_size=256, diagonal=2)


        self.base = nn.Parameter(high, requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(high), requires_grad=False)
    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

# FAD =========
class FAD_Head(nn.Module):
    def __init__(self, size, diagonal):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        # low_filter = Filter(size, 0, size // 16)
        # middle_filter = Filter(size, size // 16, size // 8)
        # high_filter = Filter(size, size // 8, size)
        # all_filter = Filter(size, 0, size * 2)
        self.diagonal = diagonal
        low_filter = LFilter(size, diagonal)
        high_filter = HFilter(size, diagonal)

        self.filters = nn.ModuleList([low_filter,high_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        # plt.imshow(x_freq[0].permute(2,1,0).cpu().detach().numpy())
        # plt.show()

        # 4 kernel
        y_list = []
        for i in range(2):
            x_pass = self.filters[i](x_freq)

            y = self._DCT_all_T @ x_pass @ self._DCT_all

            y_list.append(y)
        out = torch.cat(y_list, dim=1)
        return out


# LFS Module ======================
class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1),
                                         requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

        # init filters
        self.filters = nn.ModuleList(
            [Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i + 1), norm=True) for i in
             range(M)])

    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8) / 2) + 1
        assert size_after == 149

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)  # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            # counting mean frequency responses at a series of learnable frequency bands.
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2, 3, 4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)  # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out


# utils ===========
def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m
# or i + j <= start
def generate_filter(start, end, size):
    return [[0. if i + j > end
             else 1. for j in range(size)] for i in range(size)]

def high_filter(start, end, size):
    return [[1. if i + j > end or i + j <= start else 0. for j in range(size)] for i in range(size)]
    # end = 32, start = 256

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

def get_xcep_state_dict(pretrained_path='pretrained/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict



class Fnet(nn.Module):
    def __init__(self, img_size, diagonal):
        super(Fnet, self).__init__()
        # self.x = x
        # w, h, c = x.shape
        self.FAD_Head = FAD_Head(img_size, diagonal)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.conv1(x)
        fadx = self.FAD_Head(x)
        # lfsx = self.LFS_Head(x)

        # fea_LFS = self._norm_fea(lfsx)

        # return torch.cat((fadx, lfsx), dim=1)
        return fadx

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1,1))
        f = f.view(f.size(0), -1)
        return f

import torch
def create_filter(img_size, diagonal):
    filter = torch.ones(size=(img_size, img_size))
    filter = torch.triu(filter, diagonal=diagonal) # 0 - 256
    filter = torch.fliplr(filter)

    return filter
    # plt.imshow(filter.numpy())
    # plt.`show`()

if __name__ == '__main__':
    model = Fnet(256, diagonal = 180)

    size = 256
    src = cv2.imread('../mingi.jpg')
    src = cv2.resize(src, (256,256))
    src = np.transpose(src,(2,1,0))
    src = torch.from_numpy(np.expand_dims(src, axis=0)).float()

    out = model(src)

    # for i in range(6):
    #     out_ = out[0][i].permute(1,0).detach().cpu().numpy()
    #     plt.imshow(out_)
    #     plt.show()
