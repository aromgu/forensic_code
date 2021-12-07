import torch.nn.functional as f
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms

from models.ASPP import _ASPP
import torch
from models.RFAM import RFAM
from utils import *
from models.Unet import Unet

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
        x = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3x3(x)

        return x

class Res(nn.Module):
    def __init__(self):
        super(Res, self).__init__()

        modules = list(models.resnet18(pretrained=True).children())[:-3]
        # delete maxpooling
        del modules[3]

        self.feature_extractor = nn.Sequential(*modules) # => DenseNet
        for layer in self.feature_extractor[-1:]:
            layer.trainable = True

    def forward(self, x):
        x = f.leaky_relu(self.feature_extractor(x))
        return x

class res18autp(nn.Module):
    dim = [256, 128, 64, 32]
    def __init__(self):
        super(res18autp, self).__init__()
        # ============
        self.tp_enc = Res()

        self.unet = Unet()

        self.block1 = ASPPBlock(res18autp.dim[0], res18autp.dim[1])
        self.block2 = ASPPBlock(res18autp.dim[1], res18autp.dim[2])
        self.block3 = ASPPBlock(res18autp.dim[2], res18autp.dim[3])

        self.rfam1 = RFAM(256)

        # self.conv1x1 = nn.Conv2d(256*3, 256, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.attention = nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1,1))
        self.gap = nn.MaxPool2d(18, stride=None, padding=0, dilation=1,
                   return_indices=False, ceil_mode=False)

        # self.fcn = nn.Linear(262144, 2)

        self.out = nn.Conv2d(res18autp.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.resize = torchvision.transforms.Resize((32, 32))


    def forward(self, tp):
        # if args.train:
        reconstruct_au = self.unet(tp) # [b, 256, 32, 32]

        #     recon_feature = self.tp_enc(reconstruct_au)
        #     tp_feature = self.tp_enc(tp)
        #     ## ATTENTION
        #     cat = torch.cat((tp_feature, recon_feature), dim=1)
        #     att = self.attention(cat)
        #     gap = self.gap(att)
        #
        #     distance_size = gap.repeat(1,1,32,32)
        #     tp_fuse = self.rfam1(tp_feature, distance_size) # [b, 256, 32, 32]
        #
        #     # Decoder
        #     tp_out = self.block1(tp_fuse)
        #     tp_out = self.block2(tp_out)
        #     tp_out = self.block3(tp_out)
        #     tp_out = self.out(tp_out)
        # else:
        #     reconstruct_au = self.unet(tp)  # [b, 256, 32, 32]
        #
        #     recon_feature = self.tp_enc(reconstruct_au)
        #     tp_feature = self.tp_enc(tp)
        #
        #     ## ATTENTION
        #     cat = torch.cat((tp_feature, recon_feature), dim=1)
        #     with torch.no_grad():
        #         att = self.attention(cat)
        #     gap = self.gap(att)
        #
        #     distance_size = gap.repeat(1, 1, 32, 32)
        #     tp_fuse = self.rfam1(tp_feature, distance_size)  # [b, 256, 32, 32]
        #
        #     # Decoder
        #     tp_out = self.block1(tp_fuse)
        #     tp_out = self.block2(tp_out)
        #     tp_out = self.block3(tp_out)
        #     tp_out = self.out(tp_out)

        return reconstruct_au#, tp_out #, au_out, fcn

        # else :

if __name__ == '__main__':
    model = res18autp().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp)
