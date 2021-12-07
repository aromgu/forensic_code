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

class OneEncoder(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, dilated=True, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/.encoding/models', *args, **kwargs):
        super(OneEncoder, self).__init__()
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

        self.pretrained = get_backbone(backbone, pretrained=True, dilated=dilated,
                                       norm_layer=norm_layer, root=root,
                                       *args, **kwargs)
        self.pretrained.fc = None
        self._up_kwargs = up_kwargs
        self.upscale1 = nn.Upsample(scale_factor=2)
        self.upscale2 = nn.Upsample(scale_factor=4)

        self.bridge = bridge(512, act_fn)

        # Decoder
        self.conv1 = conv_block_2(1024, 512, act_fn)
        self.conv2 = conv_block_2(512, 256, act_fn)
        self.conv3 = conv_block_2(256, 128, act_fn)
        self.conv4 = conv_block_2(128, 64, act_fn)
        self.conv5 = conv_block_2(64, 1, act_fn)

        self.down_spl = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(2,2), padding=(1,1))

        self.onebyone1 = nn.Conv2d(256, 64, kernel_size=(1,1), stride=(1,1), dilation=1)
        self.onebyone2 = nn.Conv2d(512, 128, kernel_size=(1,1), stride=(1,1), dilation=1)
        self.onebyone3 = nn.Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1), dilation=1)
        self.onebyone4 = nn.Conv2d(2048, 512, kernel_size=(1,1), stride=(1,1), dilation=1)

        self.aspp1 = ASPPBlock(512, 512)
        self.aspp2 = ASPPBlock(256, 256)
        self.aspp3 = ASPPBlock(128, 128)
        self.aspp4 = ASPPBlock(64, 64)

        self.upscale = nn.Upsample(scale_factor=2)

        self.bn2 = torch.nn.BatchNorm2d(512)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(64)

        self.rfam4 = RFAM(512)
        self.rfam3 = RFAM(256)
        self.rfam2 = RFAM(128)
        self.rfam1 = RFAM(64)

        self.bam1 = BAM(512)
        # self.bam2 = BAM(512)
        # self.bam3 = BAM(64)

        # self.afimb1 = AFIMB(512, 128)
        # self.afimb2 = AFIMB(64, 256)

    def forward(self, rgb):

        # RGB ENCODER
        rgb = self.pretrained.conv1(rgb)
        rgb = self.pretrained.bn1(rgb)
        rgb = self.pretrained.relu(rgb)
        rgb = self.pretrained.maxpool(rgb)
        rgb1 = self.pretrained.layer1(rgb)
        rgb2 = self.pretrained.layer2(rgb1)
        rgb3 = self.pretrained.layer3(rgb2)
        rgb4 = self.pretrained.layer4(rgb3)

        rgb1 = self.onebyone1(rgb1)
        rgb2 = self.onebyone2(rgb2)
        rgb3 = self.onebyone3(rgb3)
        rgb4 = self.onebyone4(rgb4)
        rgb1 = self.upscale2(rgb1) # 256
        rgb2 = self.upscale2(rgb2) # 128
        rgb3 = self.upscale1(rgb3) # 64

        rgb4 = self.down_spl(rgb4)
        bridge_rgb = self.bridge(rgb4)
        bridge_rgb = self.aspp1(bridge_rgb)

        bridge_rgb = self.bn2(bridge_rgb)

        # DECODER
        up4 = self.bam1(bridge_rgb) # up 512

        up4 = self.bn2(up4)
        rgb4 = self.bn2(bridge_rgb)

        cat4 = torch.cat([up4, rgb4], dim=1) # 1024, 32, 32

        trans4 = self.conv1(cat4) # 1024 > 512
        trans4 = self.conv2(trans4) # 512 > 256

####################
        up3 = self.aspp2(trans4) # 256, 64, 64

        rgb3 = self.bn3(rgb3) # 256

        cat3 = torch.cat([up3, rgb3], dim=1) # 512, 64, 64

        trans3 = self.conv2(cat3) # 512 > 256
        trans3 = self.conv3(trans3) # 256 > 128

####################
        up2 = self.aspp3(trans3)  # up 256, 128, 128

        rgb2 = self.bn4(rgb2)  # 256

        cat2 = torch.cat([up2, rgb2], dim=1)  # 256

        trans2 = self.conv3(cat2)  # 256 > 128
        trans2 = self.conv4(trans2)  # 128 > 64

####################
        up1 = self.aspp4(trans2)  # up

        rgb1 = self.bn5(rgb1)  # 64

        cat1 = torch.cat([up1, rgb1], dim=1)  # 128

        trans1 = self.conv4(cat1)  # 128 > 64
        out = self.conv5(trans1)  # 64 > 1

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
    model = RFAMUPRGB(2, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16]).cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp, inp, inp)
    print(out.shape)
