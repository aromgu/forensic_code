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

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        modules = list(models.vgg16(pretrained=True).children())[:-2]

        self.feature_extractor = nn.Sequential(*modules) # => DenseNet
        for layer in self.feature_extractor[-1:]:
            layer.trainable = True

    def forward(self, x):
        x = f.leaky_relu(self.feature_extractor(x))
        return x

class VGGFeat(nn.Module):
    dim = [512, 256, 128, 64, 32]
    def __init__(self):
        super(VGGFeat, self).__init__()
        # ============
        self.enc1 = VGG()
        self.rcu1 = RCU(64, 64)
        self.rcu2 = RCU(128, 128)
        self.rcu3 = RCU(256, 256)
        self.rcu4 = RCU(512, 512)

        self.chain_pool1 = Chain_pool(32)
        self.seam1 = SEAM(VGGFeat.dim[0]*3, VGGFeat.dim[0])

        self.dec1 = nn.Sequential(
            nn.Conv2d(VGGFeat.dim[0]*2, VGGFeat.dim[1], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))
        self.dec2 = nn.Sequential(
            nn.Conv2d(VGGFeat.dim[1]*2, VGGFeat.dim[2], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))
        self.dec3 = nn.Sequential(
            nn.Conv2d(VGGFeat.dim[2]*2, VGGFeat.dim[3], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))
        self.dec4 = nn.Sequential(
            nn.Conv2d(VGGFeat.dim[3]*2, VGGFeat.dim[4], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))

        self.rfam1 = RFAM(256)
        self.rfam2 = RFAM(512)
        self.rfam3 = RFAM(512)

        self.out = nn.Conv2d(VGGFeat.dim[4], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, rgb, low, high):

        rgb1 = self.enc1.feature_extractor[0][0:5](rgb) # [b, 64, 128, 128]
        rgb2 = self.enc1.feature_extractor[0][5:10](rgb1) # [b, 128, 64, 64]
        rgb3 = self.enc1.feature_extractor[0][10:17](rgb2) # [b, 256, 32, 32]
        rgb4 = self.enc1.feature_extractor[0][17:24](rgb3) # [b, 512, 16, 16]

        low1 = self.enc1.feature_extractor[0][0:5](low) # [b, 64, 128, 128]
        low2 = self.enc1.feature_extractor[0][5:10](low1) # [b, 128, 64, 64]
        low3 = self.enc1.feature_extractor[0][10:17](low2) # [b, 256, 32, 32]
        low4 = self.enc1.feature_extractor[0][17:24](low3) # [b, 512, 16, 16]

        high1 = self.enc1.feature_extractor[0][0:5](high) # [b, 64, 128, 128]
        high2 = self.enc1.feature_extractor[0][5:10](high1) # [b, 128, 64, 64]
        high3 = self.enc1.feature_extractor[0][10:17](high2) # [b, 256, 32, 32]
        high4 = self.enc1.feature_extractor[0][17:24](high3) # [b, 512, 16, 16]

        # Refine Net ====
        # ===== RGB RCU
        rgb_rcu1 = self.rcu1(rgb1) # [b, 64, 128 ,128]
        rgb_rcu2= self.rcu2(rgb2) # [b, 128, 64, 64]
        rgb_rcu3 = self.rcu3(rgb3) # [b, 256, 32, 32]
        rgb_rcu4 = self.rcu4(rgb4) # [b, 512, 16, 16]

        # Fusion
        fuse = self.rfam2(rgb4, low4, high4) # [b, 512, 16, 16]

        # Decoder
        dec1 = self.dec1(torch.cat([fuse, rgb_rcu4], dim=1)) # [b, 256, 32, 32]
        dec2 = self.dec2(torch.cat([dec1, rgb_rcu3], dim=1)) # [b, 128, 64, 64]
        dec3 = self.dec3(torch.cat([dec2, rgb_rcu2], dim=1)) #[b, 64, 128, 128]
        dec4 = self.dec4(torch.cat([dec3, rgb_rcu1], dim=1)) #[b, 32, 256, 256]

        out = self.out(dec4)

        return out

if __name__ == '__main__':
    model = VGGFeat().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp,inp, inp)
    print(out.shape)