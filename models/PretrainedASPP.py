import torch.nn.functional as f
import torch.nn as nn
import torchvision.models as models
from models.ASPP import _ASPP
import torch

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

class ResASPP(nn.Module):
    dim = [256, 128, 64, 32]
    def __init__(self):
        super(ResASPP, self).__init__()
        # ============
        self.resnet = Res()

        self.block1 = ASPPBlock(ResASPP.dim[0], ResASPP.dim[1])
        self.block2 = ASPPBlock(ResASPP.dim[1], ResASPP.dim[2])
        self.block3 = ASPPBlock(ResASPP.dim[2], ResASPP.dim[3])

        self.out = nn.Conv2d(ResASPP.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.resnet(x) # [b, 256, 8, 8]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.out(x)

        return x

if __name__ == '__main__':
    model = ResASPP().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp)
