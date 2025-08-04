import torch.nn as nn
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.initialize import kaiming_init, constant_init


class ConvModule(nn.Module):
    """一个整合了卷积/归一化/激活层的卷积块。

    Args:
        in_channels (int): 输入特征图的通道数。与 ``nn._ConvNd`` 中的相同。
        out_channels (int): 卷积操作产生的通道数。与 ``nn._ConvNd`` 中的相同。
        kernel_size (int | tuple[int]): 卷积核的大小。与 ``nn._ConvNd`` 中的相同。
        stride (int | tuple[int]): 卷积的步长。与 ``nn._ConvNd`` 中的相同。
        padding (int | tuple[int]): 输入两侧添加的零填充。与 ``nn._ConvNd`` 中的相同。
        dilation (int | tuple[int]): 卷积核元素之间的间距。与 ``nn._ConvNd`` 中的相同。
        groups (int): 从输入通道到输出通道的分组连接数。与 ``nn._ConvNd`` 中的相同。
        bias (bool | str):是否为卷积层设置偏置项，若无归一化层，则偏置设为 True，否则为 False。默认值：False。
        with_act (bool): 是否包含非线性激活层。默认值：True。
        inplace (bool): 是否对激活使用原地模式。默认值：True。
        order (tuple[str]): 卷积/归一化/激活层的顺序。它是一个由 "conv"、"norm" 和 "act" 组成的序列。
            常见的例子有 ("conv", "norm", "act") 和 ("act", "conv", "norm")。
            默认值：('conv', 'norm', 'act')。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 with_act=True,
                 inplace=True,
                 order=('conv', 'norm', 'act')
                 ):
        super(ConvModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.with_act = with_act
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.conv = nn.Conv2d(in_channels == in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

        if order.index('norm') > order.index('conv'):
            norm_channels = out_channels
        else:
            norm_channels = in_channels
        self.norm = nn.BatchNorm2d(num_features=norm_channels, eps=1e-5)

        if self.with_act:
            self.activate = nn.ReLU6(inplace=True)

        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv, a=0, nonlinearity='relu')
        constant_init(self.norm, 1, bias=0)

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm':
                x = self.norm(x)
            elif layer == 'act' and self.with_act:
                x = self.activate(x)

        return x
