import torch.nn.functional as f
import torchvision.models as models
from models.model_core.ASPP import _ASPP
from models.model_core.RFAM import RFAM
from utils import *
from models.refine_net import RCU, Chain_pool


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

class resnext(nn.Module):
    def __init__(self):
        super(resnext, self).__init__()
        modules = list(models.resnext101_32x8d(pretrained=True).children())[:-3]
        # modules = list(models.resnet18(pretrained=True).children())[:-3]

        self.feature_extractor = nn.Sequential(*modules) # => DenseNet
        for layer in self.feature_extractor[:]:
            layer.trainable = True

    def forward(self, x):
        x = f.leaky_relu(self.feature_extractor(x))
        return x


class resnextFeat(nn.Module):
    dim = [1024, 512, 256]
    def __init__(self):
        super(resnextFeat, self).__init__()
        # ============

        self.seg = models.segmentation.deeplabv3_resnet101(pretrained=True, pretrained_backbone=True).eval()

        self.enc1 = resnext()
        self.rcu1 = RCU(256, 256)
        self.rcu2 = RCU(512, 512)
        self.rcu3 = RCU(1024, 1024)

        self.chain_pool1 = Chain_pool(256)
        self.seam1 = SEAM(resnextFeat.dim[0]*3, resnextFeat.dim[0])
        self.rfam1 = RFAM(1024)


        self.dec1 = nn.Sequential(
            nn.Conv2d(resnextFeat.dim[0]*2, resnextFeat.dim[1], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))
        self.dec2 = nn.Sequential(
            nn.Conv2d(resnextFeat.dim[1]*2, resnextFeat.dim[2], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))
        self.dec3 = nn.Sequential(
            nn.Conv2d(resnextFeat.dim[2]*2, resnextFeat.dim[2], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))

        self.out = nn.Sequential(nn.Conv2d(resnextFeat.dim[2], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.Upsample(scale_factor=2))

    def forward(self, rgb, low, high):
        rgb1 = self.enc1.feature_extractor[0:5](rgb)# [b, 256, 64, 65]
        rgb2 = self.enc1.feature_extractor[5](rgb1) # [b, 512, 32, 32]
        rgb3 = self.enc1.feature_extractor[6](rgb2) # [b, 1024, 16, 16]

        low1 = self.enc1.feature_extractor[0:5](low)# [b, 256, 64, 65]
        low2 = self.enc1.feature_extractor[5](low1) # [b, 512, 32, 32]
        low3 = self.enc1.feature_extractor[6](low2) # [b, 1024, 16, 16]

        high1 = self.enc1.feature_extractor[0:5](high)# [b, 256, 64, 65]
        high2 = self.enc1.feature_extractor[5](high1) # [b, 512, 32, 32]
        high3 = self.enc1.feature_extractor[6](high2) # [b, 1024, 16, 16]

        # Refine Net ====
        # ===== RGB RCU
        rgb_rcu1 = self.rcu1(rgb1) # [b, 256, 64 , 64]
        rgb_rcu2= self.rcu2(rgb2) # [b, 512, 32, 32]
        rgb_rcu3 = self.rcu3(rgb3) # [b, 1024, 16, 16]

        # Fusion
        fuse = self.rfam1(rgb3, low3, high3) # [b, 1024, 16, 16]

        # Decoder
        dec1 = self.dec1(torch.cat([fuse, rgb_rcu3], dim=1)) # [b, 512, 32, 32]
        dec2 = self.dec2(torch.cat([dec1, rgb_rcu2], dim=1)) #[b, 64, 128, 128]
        dec3 = self.dec3(torch.cat([dec2, rgb_rcu1], dim=1)) #[b, 32, 256, 256]

        out = self.out(dec3)

        return out

if __name__ == '__main__':
    model = resnextFeat().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp,inp, inp)
    print(out.shape)
