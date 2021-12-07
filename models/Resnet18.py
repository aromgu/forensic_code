from torchvision import models
import torch.nn as nn
from utils import *
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3_3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(3,3),stride=stride, padding=1, bias=False)

class residualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super().__init__()
        self.same_shape = same_shape
        if self.same_shape:
            stride = 1
        else:
            stride = 2
        self.conv1 = conv3_3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv3_3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=2)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = self.bn2(out)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)

class ResNet18(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channel, 64, kernel_size=[7,7], stride=2, padding=(3,3))
        self.bn0 = nn.BatchNorm2d(64)

        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            residualBlock(64, 64, True),
            residualBlock(64, 64, True)
        )

        self.block2 = nn.Sequential(
            residualBlock(64, 128, False),
            residualBlock(128, 128, True)
        )

        self.block3 = nn.Sequential(
            residualBlock(128, 256, False),
            residualBlock(256, 256, True),
        )

        self.block4 = nn.Sequential(
            residualBlock(256, 512, False),
            residualBlock(512, 512, True)
        )

        # self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.classfier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.avg(x)
        # x = x.view(x.shape[0], -1)
        # x = self.classfier(x)
        return x

def model_A(num_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    # num_features = model_resnet.fc.in_features
    # model_resnet.fc = nn.Linear(num_features, num_classes)
    del model_resnet.fc
    return model_resnet

class OurResNet(nn.Module) :
    def __init__(self):
        super(OurResNet, self).__init__()

        self.model  = model_A(num_classes=2).cuda()
        self.seq = nn.Sequential(*list(self.model.children()))

    def forward(self, x):
        out = self.seq(x)
        return out

if __name__ == '__main__':
    model = OurResNet()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp)
    print("output shape", out.shape)