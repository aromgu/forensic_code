import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Filter Module =======
class LFilter(nn.Module):
    def __init__(self, args, use_learnable=True, norm=False):
        super(LFilter, self).__init__()
        self.use_learnable = use_learnable

        # low = generate_filter(band_start, band_end, size)
        low = create_filter(img_size=args.img_size, diagonal=args.diagonal)

        self.base = nn.Parameter(low, requires_grad=True)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(args.img_size, args.img_size), requires_grad=True)
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
    def __init__(self, args, use_learnable=True, norm=False):
        super(HFilter, self).__init__()
        self.use_learnable = use_learnable

        high = create_filter(img_size=args.img_size, diagonal=args.diagonal)
        high = 1 - high

        # plt.imshow(high.cpu().detach().numpy())
        # plt.show()

        self.base = nn.Parameter(high, requires_grad=True)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(args.img_size, args.img_size), requires_grad=True)
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
    def __init__(self, args):
        super(FAD_Head, self).__init__()

        self.size = args.img_size
        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(args.img_size)).float(), requires_grad=True)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(args.img_size)).float(), 0, 1), requires_grad=True)
        # random = int(np.random.randint(256, size=1))
        random = args.diagonal
        low_filter = LFilter(args)
        high_filter = HFilter(args)

        self.filters = nn.ModuleList([low_filter,high_filter]).cuda()
    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T # [b, 3, 256, 256]
        # 4 kernel
        y_list = []
        for i in range(2):
            x_pass = self.filters[i](x_freq)
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y_list.append(y)
        out = torch.cat(y_list, dim=1)
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
    def __init__(self, args):
        super(Fnet, self).__init__()
        self.FAD_Head = FAD_Head(args)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fadx = self.FAD_Head(x)
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
    model = Fnet()

    size = 256
    src = cv2.imread('../../mingi.jpg')
    src = cv2.resize(src, (256,256))
    src = np.transpose(src,(2,1,0))
    src = torch.from_numpy(np.expand_dims(src, axis=0)).float()

    out = model(src)

    # for i in range(6):
    #     out_ = out[0][i].permute(1,0).detach().cpu().numpy()
    #     plt.imshow(out_)
    #     plt.show()
