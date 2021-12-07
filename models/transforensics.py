import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

model_urls = {
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 卷积参数变量初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # BN参数初始化
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        output['x1'] = x
        x = self.maxpool(x)

        x = self.layer1(x)
        output['x2'] = x
        x = self.layer2(x)
        output['x3'] = x
        x = self.layer3(x)
        output['x4'] = x
        x = self.layer4(x)
        output['x5'] = x

        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class transforensics(nn.Module):
    def __init__(self):
        super(transforensics, self).__init__()
        self.res = resnet50()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.pos2 = PositionalEncoding(d_hid=256, n_position=128**2)
        self.pos3 = PositionalEncoding(d_hid=256, n_position=64**2)
        self.pos4 = PositionalEncoding(d_hid=256, n_position=32**2)
        self.pos5 = PositionalEncoding(d_hid=256, n_position=16**2)
        self.conv1x1_3 = nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv1x1_4 = nn.Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv1x1_5 = nn.Conv2d(2048, 256, kernel_size=(1,1), stride=(1,1), padding=(0,0))

        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up8 = nn.Upsample(scale_factor=8)
        self.up16 = nn.Upsample(scale_factor=16)
        self.up32 = nn.Upsample(scale_factor=32)

        self.reduce_ch = nn.Conv2d(256, 1, 1)
        self.thresh = nn.Threshold(0.5, 0)

        self.out = nn.Conv2d(1,1,kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.res(x)
        x2 = x['x2']
        x3 = x['x3']
        x4 = x['x4']
        x5 = x['x5']
        one_x3 = self.conv1x1_3(x3)
        one_x4 = self.conv1x1_4(x4)
        one_x5 = self.conv1x1_5(x5)
        x2 = torch.reshape(x2, (b, -1, 256))
        x3 = torch.reshape(one_x3, (b, -1, 256))
        x4 = torch.reshape(one_x4, (b, -1, 256))
        x5 = torch.reshape(one_x5, (b, -1, 256))
        x2 = self.pos2(x2)
        x3 = self.pos3(x3)
        x4 = self.pos4(x4)
        x5 = self.pos5(x5)

        # src_mask = get_pad_mask(128**2,0)
        x2 = self.transformer_encoder(x2)
        x2 = torch.reshape(x2, (b,256,128,128))
        x2 = self.reduce_ch(x2)
        x2 = self.thresh(F.sigmoid(x2))

        x3 = self.transformer_encoder(x3)
        x3 = torch.reshape(x3, (b,256,64,64))
        x3 = self.reduce_ch(x3)
        x3 = self.thresh(F.sigmoid(x3))

        x4 = self.transformer_encoder(x4)
        x4 = torch.reshape(x4, (b,256,32,32))
        x4 = self.reduce_ch(x4)
        x4 = self.thresh(F.sigmoid(x4))

        x5 = self.transformer_encoder(x5)
        x5 = torch.reshape(x5, (b,256,16,16))
        x5 = self.reduce_ch(x5)
        x5 = self.thresh(F.sigmoid(x5))

        ## FUSE
        fuse1 = self.out(x4 * self.up2(x5)) # 32
        fuse2 = self.out(x3 * self.up2(fuse1)) # 64
        fuse3 = self.out(x2 * self.up2(fuse2)) # 128

        loss1 = self.up32(x5)
        loss2 = self.up16(fuse1)
        loss3 = self.up8(fuse2)
        loss4 = self.up4(fuse3)

        return loss1, loss2, loss3, loss4

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    img = cv2.imread('../mingi.jpg')
    img = cv2.resize(img, (512,512))
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)

    model = transforensics()
    a = np.random.rand(1, 3, 512, 512)
    a = torch.FloatTensor(a)
    o1, o2, o3, o4 = model(torch.from_numpy(img).float())
    plt.imshow(o4[0][0].cpu().detach().numpy(), cmap='gray')
    plt.show()