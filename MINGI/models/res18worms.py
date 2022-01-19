import torch
import torch.nn.functional as f
import torchvision.models as models
from models.model_core.ASPP import _ASPP
from models.model_core.RFAM import RFAM, RFAM_2
from utils import *
from models.Unet import Unet
from models.refine_net import RCU, Chain_pool
from models.model_core.CBAM import CBAM
from models.model_core.Gated_function import GATE


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

class RES_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RES_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1,1), stride=(1,1), bias=False),
            nn.ReLU())
    def forward(self, x):
        identity = self.conv(x)
        out = x + identity
        return out

class Res18worms(nn.Module):
    dim = [256, 128, 64, 32]
    def __init__(self):
        super(Res18worms, self).__init__()
        # ============
        self.enc1 = Res()
        self.unet = Unet()

        # self.dec1 = nn.Sequential(
        #     nn.Conv2d(ResFeat.dim[0]*2, ResFeat.dim[1], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.Upsample(scale_factor=2), nn.ReLU())
        # self.dec2 = nn.Sequential(
        #     nn.Conv2d(ResFeat.dim[1]*2, ResFeat.dim[2], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.Upsample(scale_factor=2), nn.ReLU())
        # self.dec3 = nn.Sequential(
        #     nn.Conv2d(ResFeat.dim[2]*2, ResFeat.dim[3], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.Upsample(scale_factor=2), nn.ReLU())

        self.dec1 = ASPPBlock(Res18worms.dim[0], Res18worms.dim[1])
        self.dec2 = ASPPBlock(Res18worms.dim[1], Res18worms.dim[2])
        self.dec3 = ASPPBlock(Res18worms.dim[2], Res18worms.dim[3])

        self.rfam1 = RFAM(256)

        self.out = nn.Conv2d(Res18worms.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, rgb, low, high):
        rgb_ = self.enc1.feature_extractor[0](rgb)
        rgb_ = self.enc1.feature_extractor[1](rgb_)
        rgb_ = self.enc1.feature_extractor[2](rgb_) # [b, 64, 128, 128]
        rgb1 = self.enc1.feature_extractor[3](rgb_) # [b, 64, 128, 128]
        rgb2 = self.enc1.feature_extractor[4](rgb1) # [b, 128, 64, 64]
        rgb3 = self.enc1.feature_extractor[5](rgb2) # [b, 256, 32, 32]

        low_ = self.enc1.feature_extractor[0](low)
        low_ = self.enc1.feature_extractor[1](low_)
        low_ = self.enc1.feature_extractor[2](low_)
        low1 = self.enc1.feature_extractor[3](low_) # [b, 64, 128, 128]
        low2 = self.enc1.feature_extractor[4](low1) # [b, 128, 64, 64]
        low3 = self.enc1.feature_extractor[5](low2) # [b, 256, 32, 32]

        high_ = self.enc1.feature_extractor[0](high)
        high_ = self.enc1.feature_extractor[1](high_)
        high_ = self.enc1.feature_extractor[2](high_)
        high1 = self.enc1.feature_extractor[3](high_) # [b, 64, 128, 128]
        high2 = self.enc1.feature_extractor[4](high1) # [b, 128, 64, 64]
        high3 = self.enc1.feature_extractor[5](high2) # [b, 256, 32, 32]

        # Fusion
        fuse = self.rfam1(rgb3, low3, high3) # [b, 256, 32, 32]

        # Decoder
        dec1 = self.dec1(fuse)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        out1 = self.out(dec3)

        unet_input = f.sigmoid(out1) * rgb

        unet_out = self.unet(unet_input)

        return out1, unet_out

class res18_base(nn.Module):
    def __init__(self):
        super(res18_base, self).__init__()
        self.pretrained_model = Res()
        self.contribution = ResFeat()

if __name__ == '__main__':
    model = ResFeat().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp,inp, inp)
