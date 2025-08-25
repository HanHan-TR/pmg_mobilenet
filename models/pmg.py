import torch.nn as nn
import torch
from typing import List

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.initialize import kaiming_init, constant_init


class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):

        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv, a=0, nonlinearity='relu')
        constant_init(self.bn, 1, bias=0)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class PMG(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 inplanes: List[int] = [256, 512, 1024],
                 feature_size: int = 512,
                 widen_factor: int = 1,
                 classes_num: int = 3) -> nn.Module:
        super(PMG, self).__init__()

        self.inplanes = [x * widen_factor for x in inplanes]
        self.feature_size = feature_size
        self.classes_num = classes_num

        self.backbone = model
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)

        self.relu = nn.ReLU(inplace=True)

        self.classifier_concat = nn.Sequential(nn.BatchNorm1d(feature_size * 3),
                                               nn.Linear(feature_size * 3, feature_size),
                                               nn.BatchNorm1d(feature_size),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(feature_size, classes_num),
                                               )

        self.conv_block1 = nn.Sequential(BasicConv(self.inplanes[0],
                                                   feature_size,
                                                   kernel_size=1, stride=1, padding=0, relu=True),
                                         BasicConv(feature_size,
                                                   feature_size,
                                                   kernel_size=3, stride=1, padding=1, relu=True)
                                         )
        self.classifier1 = nn.Sequential(nn.BatchNorm1d(feature_size),
                                         nn.Linear(feature_size, feature_size),
                                         nn.BatchNorm1d(feature_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(feature_size, classes_num),
                                         )

        self.conv_block2 = nn.Sequential(BasicConv(self.inplanes[1],
                                                   feature_size,
                                                   kernel_size=1, stride=1, padding=0, relu=True),
                                         BasicConv(feature_size,
                                                   feature_size,
                                                   kernel_size=3, stride=1, padding=1, relu=True)  # 512->1024
                                         )
        self.classifier2 = nn.Sequential(nn.BatchNorm1d(feature_size),
                                         nn.Linear(feature_size, feature_size),
                                         nn.BatchNorm1d(feature_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(feature_size, classes_num),

                                         )

        self.conv_block3 = nn.Sequential(BasicConv(self.inplanes[2],
                                                   feature_size,
                                                   kernel_size=1, stride=1, padding=0, relu=True),
                                         BasicConv(feature_size,
                                                   feature_size,
                                                   kernel_size=3, stride=1, padding=1, relu=True)
                                         )
        self.classifier3 = nn.Sequential(nn.BatchNorm1d(feature_size),
                                         nn.Linear(feature_size, feature_size),
                                         nn.BatchNorm1d(feature_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(feature_size, classes_num),
                                         )
        self.init_weights()

    def init_weights(self):
        """
        递归初始化神经网络权重
        - 对`nn.Conv2d`/`nn.Linear`使用Kaiming正态分布初始化
        - 对`nn.BatchNorm1d`/`nn.BatchNorm2d`权重初始化为1，偏置初始化为0
        - 如果模型中的子模块定义了自己的`init_weights`函数，则使用其自定义的初始化方法覆盖对其的初始化结果。
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                kaiming_init(module=module,
                             a=0,
                             mode='fan_out',
                             nonlinearity='relu',
                             bias=0,
                             distribution='normal')
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                constant_init(module, 1, bias=0)

        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights()

    def forward(self, x):
        outs = self.backbone(x)
        xf1, xf2, xf3 = outs[0], outs[1], outs[2]
        xl1 = self.conv_block1(xf1)
        xl2 = self.conv_block2(xf2)
        xl3 = self.conv_block3(xf3)

        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)

        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)
        return xc1, xc2, xc3, x_concat
