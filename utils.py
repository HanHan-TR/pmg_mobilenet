import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from typing import Union

import sys
import os
import os.path as osp
from pathlib import Path, PosixPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from models.pmg import PMG
from models.resnet import resnet50
from models.mobilenet_v2 import MobileNetV2
from core.initialize import load_state_dict


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def increment_path(work_dir: Union[str, PosixPath] = 'res',
                   project: str = 'project',
                   name: str = 'exp',
                   sep: str = '',
                   exist_ok: bool = False,  # 是否允许路径覆盖，默认不允许
                   mkdir: bool = False
                   ) -> PosixPath:
    """生成递增路径以防止命名冲突。

    当发生冲突时，自动创建顺序的目录/文件路径，
    例如 'work_dir/project/exp' -> 'work_dir/project/exp2', 'work_dir/project/exp3' 等。

    参数：
        work_dir (str/Path): 基础工作目录。默认为 'res'
        project (str): 项目子目录名称。默认为 'project'
        name (str): 实验名称。默认为 'exp'
        sep (str): 名称与递增数字之间的分隔符。默认为空字符串 ''
        exist_ok (bool): 允许覆盖现有路径。默认为 False
        mkdir (bool): 立即创建目录。默认为 False

    返回：
        Path: 生成的具有顺序递增的路径对象
    """
    path = Path(work_dir) / project / name  # Platform-independent path
    if path.exists() and not exist_ok:  # 当路径已存在且不允许覆盖时，进行递增处理
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def load_model(model_name: str,
               classes_num=3,
               widen_factor: int = 1,
               mobilenet_feature_size: int = 512,
               pretrain: bool = True,
               require_grad: bool = True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net,
                  inplanes=[256, 512, 1024],
                  feature_size=512,
                  widen_factor=1,
                  classes_num=classes_num)
    elif model_name == 'mobilenetv2_pmg':
        net = MobileNetV2(out_indices=range(0, 8), widen_factor=widen_factor)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net,
                  inplanes=[32, 96, 320],
                  feature_size=mobilenet_feature_size,
                  widen_factor=widen_factor,
                  classes_num=classes_num)
        # 加载MobileNet主干网络的预训练权值
        if pretrain:
            filename = 'pretrained_ckpt/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
            filename = osp.expanduser(filename)
            if not osp.isfile(filename):
                raise FileNotFoundError(f'{filename} can not be found.')
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            load_state_dict(net, state_dict)

    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    indices = []
    for a in range(n):
        for b in range(n):
            indices.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(indices)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = indices[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                           y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def test(net, criterion, batch_size, val_root):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.618, 0.506, 0.550), (0.265, 0.292, 0.302))
    ])
    testset = torchvision.datasets.ImageFolder(root=val_root,
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        # if use_cuda:
        #     inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            output_1, output_2, output_3, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

        if batch_idx % 10 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss
