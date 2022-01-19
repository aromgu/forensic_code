import torch
import torch.nn.functional as f
import torchvision.models as models
from models.model_core.ASPP import _ASPP
from models.model_core.RFAM import RFAM, RFAM_2
from utils import *
from models.refine_net import RCU
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
#
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
#
#     def forward(self, x):
#         x = self.aspp(x)
#         x = self.conv1x1(x)
#         x = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#         x = self.conv3x3(x)
#
#         return x
#
# class Res(nn.Module):
#     def __init__(self):
#         super(Res, self).__init__()
#
#         modules = list(models.resnet101(pretrained=True).children())[:-3]
#         # delete maxpooling
#         del modules[3]
#
#         self.feature_extractor = nn.Sequential(*modules) # => DenseNet
#         for layer in self.feature_extractor[-1:]:
#             layer.trainable = True
#
#     def forward(self, x):
#         x = f.leaky_relu(self.feature_extractor(x))
#         return x
#
# class RES_Block(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(RES_Block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, kernel_size=(1,1), stride=(1,1), bias=False),
#             nn.ReLU())
#     def forward(self, x):
#         identity = self.conv(x)
#         out = x + identity
#         return out
#
# class Res101(nn.Module):
#     dim = [512, 256, 128, 64]
#     def __init__(self):
#         super(Res101, self).__init__()
#         # ============
#         self.enc1 = Res()
#         self.rcu1 = RCU(128, 128)
#         self.rcu2 = RCU(256, 256)
#         self.rcu3 = RCU(512, 512)
#
#         self.dec1 = ASPPBlock(Res101.dim[0]*2 + 21, Res101.dim[1])
#         self.dec2 = ASPPBlock(Res101.dim[1]*2 + 21, Res101.dim[2])
#         self.dec3 = ASPPBlock(Res101.dim[2]*2 + 21, Res101.dim[3])
#         # self.dec1 = nn.Sequential(
#         #     nn.Conv2d(Res101.dim[0]*2 + 21, Res101.dim[1], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#         #     nn.Upsample(scale_factor=2), nn.ReLU())
#         # self.dec2 = nn.Sequential(
#         #     nn.Conv2d(Res101.dim[1]*2 + 21, Res101.dim[2], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#         #     nn.Upsample(scale_factor=2), nn.ReLU())
#         # self.dec3 = nn.Sequential(
#         #     nn.Conv2d(Res101.dim[2]*2 + 21, Res101.dim[3], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#         #     nn.Upsample(scale_factor=2), nn.ReLU())
#
#         self.rfam1 = RFAM(512)
#         # self.rfam_dec1 = RFAM_2(256,256)
#         # self.rfam_dec2 = RFAM_2(128,128)
#         # self.rfam_dec3 = RFAM_2(64,64)
#
#         self.one_1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
#         self.one_2 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
#         self.one_3 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
#
#         self.fuse_add = nn.Sequential(nn.Conv2d(256, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#                                       nn.Upsample(scale_factor=8))
#
#         self.low_level1 = nn.Conv2d(3, 128, kernel_size=(4,4), stride=(4,4))
#         self.low_level2 = nn.Conv2d(3, 64, kernel_size=(2,2), stride=(2,2))
#         self.low_level3 = nn.Conv2d(3, 32, kernel_size=(1,1), stride=(1,1))
#
#         self.seg = models.segmentation.deeplabv3_resnet101(pretrained=True, pretrained_backbone=True).eval()
#         self.seg_scale1 = nn.Conv2d(21,21, kernel_size=(8,8), stride=(8,8))
#         self.seg_scale2 = nn.Conv2d(21,21, kernel_size=(4,4), stride=(4,4))
#         self.seg_scale3 = nn.Conv2d(21,21, kernel_size=(2,2), stride=(2,2))
#
#         self.out = nn.Conv2d(Res101.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#
#     def forward(self, rgb, low, high):
#         rgb_ = self.enc1.feature_extractor[0](rgb)
#         rgb_ = self.enc1.feature_extractor[1](rgb_)
#         rgb_ = self.enc1.feature_extractor[2](rgb_) # [b, 256, 128, 128]
#         rgb1 = self.enc1.feature_extractor[3](rgb_) # [b, 256, 128, 128]
#         rgb2 = self.enc1.feature_extractor[4](rgb1) # [b, 512, 64, 64]
#         rgb3 = self.enc1.feature_extractor[5](rgb2) # [b, 1024, 32, 32]
#
#         low_ = self.enc1.feature_extractor[0](low)
#         low_ = self.enc1.feature_extractor[1](low_)
#         low_ = self.enc1.feature_extractor[2](low_)
#         low1 = self.enc1.feature_extractor[3](low_) # [b, 256, 128, 128]
#         low2 = self.enc1.feature_extractor[4](low1) # [b, 512, 64, 64]
#         low3 = self.enc1.feature_extractor[5](low2) # [b, 1024, 32, 32]
#
#         high_ = self.enc1.feature_extractor[0](high)
#         high_ = self.enc1.feature_extractor[1](high_)
#         high_ = self.enc1.feature_extractor[2](high_)
#         high1 = self.enc1.feature_extractor[3](high_) # [b, 256, 128, 128]
#         high2 = self.enc1.feature_extractor[4](high1) # [b, 512, 64, 64]
#         high3 = self.enc1.feature_extractor[5](high2) # [b, 1024, 32, 32]
#
#         # Refine Net ====
#         # ===== RGB RCU
#         rgb1_ = self.one_2(rgb1)
#         rgb_rcu1 = self.rcu1(rgb1_) # [b, 128, 128 ,128]
#
#         rgb2_ = self.one_3(rgb2)
#         rgb_rcu2= self.rcu2(rgb2_) # [b, 256, 64, 64]
#
#         rgb3_ = self.one_1(rgb3)
#         rgb_rcu3 = self.rcu3(rgb3_) # [b, 512, 32, 32]
#
#         one_rgb = self.one_1(rgb3)
#         one_low = self.one_1(low3)
#         one_high = self.one_1(high3)
#
#         # Fusion
#         fuse = self.rfam1(one_rgb, one_low, one_high) # [b, 512, 32, 32]
#
#         seg_out = self.seg(rgb) # out, aux, [b, 21, 256, 256]
#         seg_out = seg_out['out']
#         scale1 = self.seg_scale1(seg_out) # [b, 21, 32, 32]
#         scale2 = self.seg_scale2(seg_out) # [b, 21, 64, 64]
#         scale3 = self.seg_scale3(seg_out) # [b, 21, 128, 128]
#
#         # Decoder
#         dec1 = self.dec1(torch.cat([fuse, rgb_rcu3, scale1], dim=1))# [b, 256, 64, 64]
#         dec2 = self.dec2(torch.cat([dec1, rgb_rcu2, scale2], dim=1))# [b, 128, 128, 128]
#         dec3 = self.dec3(torch.cat([dec2, rgb_rcu1, scale3], dim=1))#[b, 64, 256, 256]
#
#         # dec1 = self.dec1(torch.cat([fuse, rgb_rcu3, scale1], dim=1))# [b, 256, 64, 64]
#         # dec2 = self.dec2(torch.cat([dec1, rgb_rcu2, scale2], dim=1))# [b, 128, 128, 128]
#         # dec3 = self.dec3(torch.cat([dec2, rgb_rcu1, scale3], dim=1))#[b, 64, 256, 256]
#
#         out = self.out(dec3)
#
#         return out
#
# if __name__ == '__main__':
#     model = Res101().cuda()
#     inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
#     out = model(inp,inp, inp)


# import torch.nn.functional as f
# import torchvision.models as models
# from models.ASPP import _ASPP
# from models.model_core.RFAM import RFAM
# from utils import *
# from models.refine_net import RCU
# import segmentation_models as sm
# sm.set_framework('tf.keras')
# sm.framework()
# import torch

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

        modules = list(models.resnet101(pretrained=True).children())[:-3]
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

class Res101(nn.Module):
    dim = [512, 256, 128, 64]
    def __init__(self):
        super(Res101, self).__init__()
        # ============
        self.enc1 = Res()
        self.rcu1 = RCU(128, 128)
        self.rcu2 = RCU(256, 256)
        self.rcu3 = RCU(512, 512)

        self.dec1 = nn.Sequential(
            nn.Conv2d(Res101.dim[0]*2 + 21, Res101.dim[1], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))
        self.dec2 = nn.Sequential(
            nn.Conv2d(Res101.dim[1]*2 + 21, Res101.dim[2], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))
        self.dec3 = nn.Sequential(
            nn.Conv2d(Res101.dim[2]*2 + 21, Res101.dim[3], kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=2))

        self.rfam1 = RFAM(512)

        self.rfam_dec1 = RFAM_2(256,256)
        self.rfam_dec2 = RFAM_2(128,128)
        self.rfam_dec3 = RFAM_2(64,64)

        self.one_1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
        self.one_2 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
        self.one_3 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))

        self.fuse_add = nn.Sequential(nn.Conv2d(256, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                      nn.Upsample(scale_factor=8))

        self.low_level1 = nn.Conv2d(3, 128, kernel_size=(4,4), stride=(4,4))
        self.low_level2 = nn.Conv2d(3, 64, kernel_size=(2,2), stride=(2,2))
        self.low_level3 = nn.Conv2d(3, 32, kernel_size=(1,1), stride=(1,1))

        self.seg = models.segmentation.deeplabv3_resnet101(pretrained=True, pretrained_backbone=True).eval()
        self.seg_scale1 = nn.Conv2d(21,21, kernel_size=(8,8), stride=(8,8))
        self.seg_scale2 = nn.Conv2d(21,21, kernel_size=(4,4), stride=(4,4))
        self.seg_scale3 = nn.Conv2d(21,21, kernel_size=(2,2), stride=(2,2))

        self.out = nn.Conv2d(Res101.dim[3], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, rgb, low, high):
        rgb_ = self.enc1.feature_extractor[0](rgb)
        rgb_ = self.enc1.feature_extractor[1](rgb_)
        rgb_ = self.enc1.feature_extractor[2](rgb_) # [b, 256, 128, 128]
        rgb1 = self.enc1.feature_extractor[3](rgb_) # [b, 256, 128, 128]
        rgb2 = self.enc1.feature_extractor[4](rgb1) # [b, 512, 64, 64]
        rgb3 = self.enc1.feature_extractor[5](rgb2) # [b, 1024, 32, 32]

        low_ = self.enc1.feature_extractor[0](low)
        low_ = self.enc1.feature_extractor[1](low_)
        low_ = self.enc1.feature_extractor[2](low_)
        low1 = self.enc1.feature_extractor[3](low_) # [b, 256, 128, 128]
        low2 = self.enc1.feature_extractor[4](low1) # [b, 512, 64, 64]
        low3 = self.enc1.feature_extractor[5](low2) # [b, 1024, 32, 32]

        high_ = self.enc1.feature_extractor[0](high)
        high_ = self.enc1.feature_extractor[1](high_)
        high_ = self.enc1.feature_extractor[2](high_)
        high1 = self.enc1.feature_extractor[3](high_) # [b, 256, 128, 128]
        high2 = self.enc1.feature_extractor[4](high1) # [b, 512, 64, 64]
        high3 = self.enc1.feature_extractor[5](high2) # [b, 1024, 32, 32]

        # Refine Net ====
        # ===== RGB RCU
        rgb1_ = self.one_2(rgb1)
        rgb_rcu1 = self.rcu1(rgb1_) # [b, 128, 128 ,128]

        rgb2_ = self.one_3(rgb2)
        rgb_rcu2= self.rcu2(rgb2_) # [b, 256, 64, 64]

        rgb3_ = self.one_1(rgb3)
        rgb_rcu3 = self.rcu3(rgb3_) # [b, 512, 32, 32]

        one_rgb = self.one_1(rgb3)
        one_low = self.one_1(low3)
        one_high = self.one_1(high3)

        # Fusion
        fuse = self.rfam1(one_rgb, one_low, one_high) # [b, 512, 32, 32]

        seg_out = self.seg(rgb) # out, aux, [b, 21, 256, 256]
        seg_out = seg_out['out']
        scale1 = self.seg_scale1(seg_out) # [b, 21, 32, 32]
        scale2 = self.seg_scale2(seg_out) # [b, 21, 64, 64]
        scale3 = self.seg_scale3(seg_out) # [b, 21, 128, 128]

        # Decoder
        dec1 = self.rfam_dec1(self.dec1(torch.cat([fuse, rgb_rcu3, scale1], dim=1)))# [b, 256, 64, 64]
        dec2 = self.rfam_dec2(self.dec2(torch.cat([dec1, rgb_rcu2, scale2], dim=1)))# [b, 128, 128, 128]
        dec3 = self.rfam_dec3(self.dec3(torch.cat([dec2, rgb_rcu1, scale3], dim=1)))#[b, 64, 256, 256]

        out = self.out(dec3)

        return out

if __name__ == '__main__':
    model = Res101().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp,inp, inp)