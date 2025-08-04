# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from models.base.conv_module import ConvModule
from core.math import make_divisible


class InvertedResidual(nn.Module):
    """MobileNetV2的反向残差块。

    Args:
        in_channels (int): 反向残差块的输入通道数。
        out_channels (int): 反向残差块的输出通道数。
        stride (int): 中间（第一个）3x3卷积的步长。
        expand_ratio (int): 通过此值调整反向残差块中隐藏层的通道数。
        conv_cfg (dict, 可选): 卷积层的配置字典。
            默认值: None，表示使用conv2d。
        norm_cfg (dict): 归一化层的配置字典。
            默认值: dict(type='BN')。
        act_cfg (dict): 激活层的配置字典。
            默认值: dict(type='ReLU6')。
        with_cp (bool): 是否使用检查点。使用检查点会节省一些内存，但会降低训练速度。默认值: False。

    Returns:
        Tensor: 输出张量
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 with_cp=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels=in_channels,
                                     out_channels=hidden_dim,
                                     kernel_size=1))

        layers.extend([ConvModule(in_channels=hidden_dim,
                                  out_channels=hidden_dim,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1,
                                  groups=hidden_dim),
                       ConvModule(in_channels=hidden_dim,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  with_act=False)
                       ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class MobileNetV2(nn.Module):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1],  # 0 layer1: 1/2
                     [6, 24, 2, 2],  # 1 layer2: 1/2 xf1
                     [6, 32, 3, 2],  # 2 layer3: 1/4 xf2
                     [6, 64, 4, 2],  # 3 layer4: 1/8 xf3
                     [6, 96, 3, 1],  # 4 layer5: 1/16
                     [6, 160, 3, 2],  # 5 layer6: 1/16 xf4
                     [6, 320, 1, 1]  # 6 layer7: 1/32 xf5
                     ]

    def __init__(self,
                 widen_factor=1.,
                 out_indices=(1, 2, 3, 5, 6),
                 with_cp=False):
        super(MobileNetV2, self).__init__()
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')
        self.with_cp = with_cp

        self.in_channels = make_divisible(32 * widen_factor, 8)

        self.conv1 = ConvModule(in_channels=3,
                                out_channels=self.in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(out_channels=out_channels,
                                                 num_blocks=num_blocks,
                                                 stride=stride,
                                                 expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(1280 * widen_factor)
        else:
            self.out_channel = 1280

        layer = ConvModule(in_channels=self.in_channels,
                           out_channels=self.out_channel,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(InvertedResidual(self.in_channels,
                                           out_channels,
                                           stride,
                                           expand_ratio=expand_ratio,
                                           with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs[0], outs[1], outs[2], outs[3], outs[4]


if __name__ == "__main__":
    model = MobileNetV2()
    pass
