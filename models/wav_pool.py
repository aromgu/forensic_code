import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import cv2

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        # B, CH, H, W
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2

        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        xll = x1 + x2 + x3 + x4
        xlh = - x1 + x2 - x3 + x4
        xhl = - x1 - x2 + x3 + x4
        xhh = x1 - x2 - x3 + x4

        return xll, xhl, xlh, xhh
    # ll = low pass filter


class IWT(nn.Module):
    def __init__(self, cuda_flag=True):
        super().__init__()
        self.requires_grad = False
        self.cuda_flag = cuda_flag

    def forward(self, x):
        r = 2
        # x = torch.cat((x1,x2,x3,x4), 1)
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = \
            in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()
        if self.cuda_flag: h = h.cuda()

        # 0부터 끝까지 2씩 이동
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

if __name__ == "__main__":
# DWT =================================
    # B, CH, H, W

    # x = np.ones((1,4,100,100))
    x = cv2.imread(('./img1.jpg'),0)
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)

    # /2 는 내부 값이 반이 됨 | shape (:,:,반,:)
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2

    # shape : (:,:,반,반)
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    xll = x1 + x2 + x3 + x4
    xlh = - x1 + x2 - x3 + x4
    xhl = - x1 - x2 + x3 + x4
    xhh = x1 - x2 - x3 + x4

    xll1 = np.transpose(xll[0], (1, 2, 0))
    xlh1 = np.transpose(xlh[0], (1, 2, 0))
    xhl1 = np.transpose(xhl[0], (1 ,2, 0))
    xhh1 = np.transpose(xhh[0], (1, 2, 0))

    x = np.concatenate((xll, xlh, xhl, xhh), axis=1)
    fig, ax = plt.subplots(2,2)
    ax[0][0].imshow(xll1, cmap='gray')
    ax[0][1].imshow(xlh1, cmap='gray')
    ax[1][0].imshow(xhl1, cmap='gray')
    ax[1][1].imshow(xhh1, cmap='gray')
    plt.show()

# IWT ============================

    r = 2
    (in_batch, in_channel, in_height, in_width) = x.shape

    # out_channel = in_channel/4
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = np.zeros([out_batch, out_channel, out_height, out_width])

    # 0부터 끝까지 2씩 이동
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    iimg = h[0]
    iimg = np.transpose(iimg, (1,2,0))
    plt.imshow(iimg, cmap='gray')
    plt.show()
    print(iimg.shape)

# DWT
# l1 = nn.Conv2d(in_ch1, out_ch1, ker_size, stride=2)
# l2 = nn.Conv2d(in_ch2, out_ch2, ker_size)
# ------>
# l1 = nn.Conv2d(in_ch, out_ch, ker_size, stride=1)
# wp = DWT()
# l2 = nn.Conv2d(in_ch2*2, out_ch, ker_size)