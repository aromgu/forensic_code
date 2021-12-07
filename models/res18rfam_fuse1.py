import torch.nn.functional as f
import torch.nn as nn
import torchvision.models as models
from models.ASPP import _ASPP
import torch
from models.RFAM import RFAM

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

class Res18rfam(nn.Module):
    dim = [256, 128, 64, 32]
    def __init__(self):
        super(Res18rfam, self).__init__()
        # ============
        self.enc1 = Res()

        self.block1 = ASPPBlock(Res18rfam.dim[0], Res18rfam.dim[1])
        # self.block1 = ASPPBlock(128, Res18rfam.dim[1])
        self.block2 = ASPPBlock(Res18rfam.dim[1], Res18rfam.dim[2])
        self.block3 = ASPPBlock(Res18rfam.dim[2], Res18rfam.dim[3])

        self.rfam1 = RFAM(256)

        # self.conv1x1 = nn.Conv2d(256*3, 256, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))

        self.out = nn.Conv2d(Res18rfam.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, rgb, low, high):
        rgb = self.enc1(rgb) # [b, 256, 32, 32]
        low = self.enc1(low)
        high = self.enc1(high)

        # Fusion
        # fuse1 =
        fuse = self.rfam1(rgb, low, high) # [b, 256, 32, 32]
        # cat = torch.cat((rgb,low,high), dim=1)
        # one = self.conv1x1(cat)

        # Decoder
        dec = self.block1(fuse)
        dec = self.block2(dec)
        dec = self.block3(dec)
        out = self.out(dec)

        return out

if __name__ == '__main__':
    model = Res18rfam().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp,inp,inp)
