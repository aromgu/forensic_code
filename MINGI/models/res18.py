import torch
import torch.nn.functional as f
import torchvision.models as models
from models.model_core.ASPP import _ASPP
from models.model_core.RFAM import RFAM, RFAM_2
from utils import *
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

# class ASPPBlock(nn.Module) :
#     def __init__(self, in_ch, out_ch):
#         super(ASPPBlock, self).__init__()
#
#         self.aspp = _ASPP(in_ch, in_ch, [1,6,12,18])
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(4 * in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
#             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch)
#         )
#         self.conv3x3 = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch)
#         )
#         self.aspp.apply(self.init_weights)
#         self.conv1x1.apply(self.init_weights)
#         self.conv3x3.apply(self.init_weights)
#
#     def init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             torch.nn.init.xavier_uniform(m.weight)
#             # torch.nn.init.normal_(m.weight, mean=0, std=0.02)
#             # m.bias.data.fill_(0.01)

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

class ResFeat(nn.Module):
    dim = [256, 128, 64, 32]
    def __init__(self):
        super(ResFeat, self).__init__()
        # ============
        # self.bn = nn.BatchNorm2d(3)
        self.enc1 = Res()
        #self.unet = Unet()
        self.rcu1 = RCU(64, 64)
        self.rcu2 = RCU(128, 128)
        self.rcu3 = RCU(256, 256)

        self.chain_pool1 = Chain_pool(32)

        self.dec1 = nn.Sequential(
            nn.Conv2d(ResFeat.dim[0]*2 + 21, ResFeat.dim[1], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2), nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.Conv2d(ResFeat.dim[1]*2 + 21, ResFeat.dim[2], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2), nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.Conv2d(ResFeat.dim[2]*2 + 21, ResFeat.dim[3], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2), nn.ReLU())

        # self.dec1 = ASPPBlock(ResFeat.dim[0], ResFeat.dim[1])
        # self.dec2 = ASPPBlock(ResFeat.dim[1], ResFeat.dim[2])
        # self.dec3 = ASPPBlock(ResFeat.dim[2], ResFeat.dim[3])

        self.rfam1 = RFAM(256)
        self.rfam_dec1 = RFAM_2(128,128)
        self.rfam_dec2 = RFAM_2(64,64)
        self.rfam_dec3 = RFAM_2(32,32)

        # self.gay1 = GATE(128, 64)
        # self.gay2 = GATE(64, 128)
        # self.gay3 = GATE(32, 256)

        self.conv1x1 = nn.Conv2d(256*3, 256, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))

        # self.fuse_add = nn.Sequential(nn.Conv2d(256, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #                               nn.Upsample(scale_factor=8))

        # self.seam1 = SELayer(128)
        # self.seam2 = SELayer(64)
        # self.seam3 = SELayer(32)

        # self.low_level1 = nn.Conv2d(3, 128, kernel_size=(4,4), stride=(4,4))
        # self.low_level2 = nn.Conv2d(3, 64, kernel_size=(2,2), stride=(2,2))
        # self.low_level3 = nn.Conv2d(3, 32, kernel_size=(1,1), stride=(1,1))

        self.seg = models.segmentation.deeplabv3_resnet101(pretrained=True, pretrained_backbone=True).eval()
        self.seg_scale2 = nn.Conv2d(21,21, kernel_size=(8,8), stride=(8,8))
        self.seg_scale3 = nn.Conv2d(21,21, kernel_size=(4,4), stride=(4,4))
        self.seg_scale4 = nn.Conv2d(21,21, kernel_size=(2,2), stride=(2,2))

        self.out = nn.Conv2d(ResFeat.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.out_edge = nn.Conv2d(ResFeat.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

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

        # Refine Net ====
        rgb_rcu1 = self.rcu1(rgb1) # [b, 64, 128 ,128]
        rgb_rcu2= self.rcu2(rgb2) # [b, 128, 64, 64]
        rgb_rcu3 = self.rcu3(rgb3) # [b, 256, 32, 32]

        # Fusion
        fuse = self.rfam1(rgb3, low3, high3) # [b, 256, 32, 32]

        seg = self.seg(rgb) # out, aux, [2, 21, 256, 256]
        seg = seg['out']

        scale1 = self.seg_scale2(seg) # [b, 21, 32, 32]
        scale2 = self.seg_scale3(seg) # [b, 21, 64, 64]
        scale3 = self.seg_scale4(seg) # [b, 21, 64, 64] 됬담

        # Decoder # 더 할 꺼 있엉?
        # dec1 = self.dec1(fuse)
        # dec2 = self.dec2(dec1)
        # dec3 = self.dec3(dec2)
        # dec1 = self.dec1(torch.cat([fuse, rgb3], dim=1)) # [b, 128, 64, 64]
        # dec2 = self.dec2(torch.cat([dec1, rgb2], dim=1)) # [b, 64, 128, 128]
        # dec3 = self.dec3(torch.cat([dec2, rgb1], dim=1))# + low_level) # [b, 32, 256, 256] # fuse, high add fuse
        dec1 = f.dropout(self.rfam_dec1(self.dec1(torch.cat([fuse, rgb_rcu3, scale1], dim=1))), p=0.5) # [b, 128, 64, 64]
        dec2 = f.dropout(self.rfam_dec2(self.dec2(torch.cat([dec1, rgb_rcu2, scale2], dim=1))), p=0.5) # [b, 64, 128, 128]
        dec3 = f.dropout(self.rfam_dec3(self.dec3(torch.cat([dec2, rgb_rcu1, scale3], dim=1))), p=0.5)
        # [b, 32, 256, 256]

        out = self.out(dec3)
        # out_edge = self.out_edge(dec3)

        return out

class res18_base(nn.Module):
    def __init__(self):
        super(res18_base, self).__init__()
        self.pretrained_model = Res()
        self.contribution = ResFeat()

if __name__ == '__main__':
    model = ResFeat().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp,inp, inp)
