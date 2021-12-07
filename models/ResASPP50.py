# https://github.com/junfu1115/DANet

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter
from models.DualAttention import BAM
from utils.DFMBFI import AFIMB

# from ...utils import batch_pix_accuracy, batch_intersection_union

from models.resnet import *
from models.RFAM import RFAM

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet']

def get_backbone(name, **kwargs):
    models = {
        # resnet
        'resnet50': resnet50,
        # 'resnet101': resnet101,
        # 'resnet152': resnet152,
        # # resnest
        # 'resnest50': resnest50,
        # 'resnest101': resnest101,
        # 'resnest200': resnest200,
        # 'resnest269': resnest269,
        # # resnet other variants
        # 'resnet50s': resnet50s,
        # 'resnet101s': resnet101s,
        # 'resnet152s': resnet152s,
        # 'resnet50d': resnet50d,
        # 'resnext50_32x4d': resnext50_32x4d,
        # 'resnext101_32x8d': resnext101_32x8d,
        # # other segmentation backbones
        # 'xception65': xception65,
        # 'wideresnet38': wideresnet38,
        # 'wideresnet50': wideresnet50,
        }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net

def conv_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model

def bridge(in_dim, act_fn) :
    model = nn.Sequential(
        nn.Conv2d(in_dim, in_dim*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Conv2d(in_dim*2, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        act_fn
    )
    return model

def conv_block_2(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim)
    )

    return model

def conv_trans_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        nn.BatchNorm2d(out_dim), act_fn
    )

    return model


from models.ASPP import _ASPP

class ASPPBlock(nn.Module) :
    def __init__(self, in_ch, out_ch):
        super(ASPPBlock, self).__init__()
        self.aspp = _ASPP(in_ch, in_ch, [1,6,12,18])
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(4 * in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.conv1x1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv3x3(x)

        return x

class ResASPP50(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, dilated=True, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/.encoding/models', *args, **kwargs):
        super(ResASPP50, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        self.backbone = backbone
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        # self.pretrained = get_backbone(backbone, pretrained=True, dilated=dilated,
        #                                norm_layer=norm_layer, root=root,
        #                                *args, **kwargs)
        modules = list(get_backbone(backbone, pretrained=True, dilated=dilated,
                                       norm_layer=norm_layer, root=root,
                                       *args, **kwargs).children())[:-2]

        self.feature_extractor = nn.Sequential(*modules)
        for layer in self.feature_extractor[-1]:
            layer.trainable = True

        self._up_kwargs = up_kwargs

        self.bridge = bridge(512, act_fn)

        # Decoder
        self.conv1 = conv_block_2(1024, 512, act_fn)
        self.conv2 = conv_block_2(512, 256, act_fn)
        self.conv3 = conv_block_2(256, 128, act_fn)
        self.conv4 = conv_block_2(128, 64, act_fn)
        self.conv5 = conv_block_2(64, 1, act_fn)

        self.aspp1 = ASPPBlock(2048, 512)
        self.aspp2 = ASPPBlock(512, 128)
        self.aspp3 = ASPPBlock(128, 1)

    def forward(self, rgb):

        # RGB ENCODER
        rgb = self.feature_extractor[0](rgb)
        rgb = self.feature_extractor[1](rgb)
        rgb = self.feature_extractor[2](rgb)
        rgb = self.feature_extractor[3](rgb)
        rgb1 = self.feature_extractor[4](rgb)
        rgb2 = self.feature_extractor[5](rgb1)
        rgb3 = self.feature_extractor[6](rgb2)
        rgb4 = self.feature_extractor[7](rgb3) # [b, 2048, 32, 32]

        # Decoder
        x = self.aspp1(rgb4)
        x = self.aspp2(x)
        out = self.aspp3(x)

        return out

def module_inference(module, image, flip=True):
    output = module.evaluate(image)
    if flip:
        fimg = flip_image(image)
        foutput = module.evaluate(fimg)
        output += flip_image(foutput)
    return output.exp()

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)

if __name__ == '__main__':
    model = ResASPP50(2, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16]).cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp)
    print('out', out.shape)
