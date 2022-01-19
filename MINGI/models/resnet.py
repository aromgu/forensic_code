##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNet variants"""
import math
import torch
import torch.nn as nn

# from ...nn import DropBlock2D
from models.model_store import get_model_file

__all__ = ['ResNet', 'Bottleneck',
           'resnet50', 'resnet101', 'resnet152']

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size, share_channel=False):
        super(DropBlock2D, self).__init__()
        self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
        self.register_buffer('drop_prob', drop_prob * torch.ones(1, dtype=torch.float32))
        self.inited = False
        self.step_size = 0.0
        self.start_step = 0
        self.nr_steps = 0
        self.block_size = block_size
        self.share_channel = share_channel

    def reset(self):
        """stop DropBlock"""
        self.inited = True
        self.i[0] = 0
        self.drop_prob = 0.0

    def reset_steps(self, start_step, nr_steps, start_value=0, stop_value=None):
        self.inited = True
        stop_value = self.drop_prob.item() if stop_value is None else stop_value
        self.i[0] = 0
        self.drop_prob[0] = start_value
        self.step_size = (stop_value - start_value) / nr_steps
        self.nr_steps = nr_steps
        self.start_step = start_step

    def forward(self, x):
        if not self.training or self.drop_prob.item() == 0.:
            return x
        else:
            self.step()

            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask and place on input device
            if self.share_channel:
                mask = (torch.rand(*x.shape[2:], device=x.device, dtype=x.dtype) < gamma).unsqueeze(0).unsqueeze(0)
            else:
                mask = (torch.rand(*x.shape[1:], device=x.device, dtype=x.dtype) < gamma).unsqueeze(0)

            # compute block mask
            block_mask, keeped = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * (block_mask.numel() / keeped).to(out)
            return out


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        # self.bn1 = norm_layer(group_width)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        self.conv2 = nn.Conv2d(
            group_width, group_width, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, bias=False)
        self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d,
                 multi_grid=True, multi_dilation=[4, 8, 16], os=8, no_deepstem=False):
        if no_deepstem:
            deep_stem=False
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        # if rectified_conv:
        #     conv_layer = RFConv2d
        # else:
        conv_layer = nn.Conv2d

        if os ==8:
            dilation=4
        elif os==16:
            dilation=2

        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                # norm_layer(stem_width),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                # norm_layer(stem_width),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        # self.bn1 = norm_layer(self.inplanes)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=nn.BatchNorm2d, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=nn.BatchNorm2d)

        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=nn.BatchNorm2d,
                                           dropblock_prob=dropblock_prob)
            if not multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=nn.BatchNorm2d,
                                               dropblock_prob=dropblock_prob)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                               dilation=4, norm_layer=nn.BatchNorm2d,
                               dropblock_prob=dropblock_prob, multi_grid=True, multi_dilation=multi_dilation)
        elif dilation==2:
            assert multi_grid is not True, "multi_grid implement when os 8"
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=nn.BatchNorm2d,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=nn.BatchNorm2d,
                                           dropblock_prob=dropblock_prob)
        else:
            assert multi_grid is not True, "multi_grid implement when os 8"
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=nn.BatchNorm2d,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=nn.BatchNorm2d,
                                           dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True, multi_grid=True, multi_dilation=[4,8,16]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []

        ## for multi_grid
        if not multi_grid:
            if dilation == 1 or dilation == 2:
                layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                    radix=self.radix, cardinality=self.cardinality,
                                    bottleneck_width=self.bottleneck_width,
                                    avd=self.avd, avd_first=self.avd_first,
                                    dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                    rectify_avg=self.rectify_avg,
                                    norm_layer=nn.BatchNorm2d, dropblock_prob=dropblock_prob,
                                    last_gamma=self.last_gamma))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                    radix=self.radix, cardinality=self.cardinality,
                                    bottleneck_width=self.bottleneck_width,
                                    avd=self.avd, avd_first=self.avd_first,
                                    dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                    rectify_avg=self.rectify_avg,
                                    norm_layer=nn.BatchNorm2d, dropblock_prob=dropblock_prob,
                                    last_gamma=self.last_gamma))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        else:
            assert multi_dilation is not None, "please set the multi_dilation correctly"

            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                    radix=self.radix, cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd, avd_first=self.avd_first,
                    dilation=multi_dilation[0], is_first=is_first, rectified_conv=self.rectified_conv,
                    rectify_avg=self.rectify_avg,
                    norm_layer=nn.BatchNorm2d, dropblock_prob=dropblock_prob,
                    last_gamma=self.last_gamma))

        self.inplanes = planes * block.expansion

        ## for multi_grid
        if not multi_grid:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes,
                                    radix=self.radix, cardinality=self.cardinality,
                                    bottleneck_width=self.bottleneck_width,
                                    avd=self.avd, avd_first=self.avd_first,
                                    dilation=dilation, rectified_conv=self.rectified_conv,
                                    rectify_avg=self.rectify_avg,
                                    norm_layer=nn.BatchNorm2d, dropblock_prob=dropblock_prob,
                                    last_gamma=self.last_gamma))
        else:
            assert multi_dilation is not None, "please set the multi_dilation correctly"
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes,
                                    radix=self.radix, cardinality=self.cardinality,
                                    bottleneck_width=self.bottleneck_width,
                                    avd=self.avd, avd_first=self.avd_first,
                                    dilation=multi_dilation[i%div], rectified_conv=self.rectified_conv,
                                    rectify_avg=self.rectify_avg,
                                    norm_layer=nn.BatchNorm2d, dropblock_prob=dropblock_prob,
                                    last_gamma=self.last_gamma))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x

def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50', root=root)), strict=False)
    return model


def resnet101(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    pretrained=False
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet101', root=root)), strict=False)
    return model


def resnet152(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet152', root=root)), strict=False)
    return model