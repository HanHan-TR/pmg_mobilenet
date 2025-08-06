from __future__ import print_function
import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models.pmg import PMG
from models.resnet import resnet50
from utils import jigsaw_generator, load_model, cosine_anneal_schedule, increment_path, test


def train(train_root: str,
          val_root: str,
          model_name: str,
          classes_num=3,
          widen_factor: int = 1,
          mobilenet_feature_size: int = 512,
          pretrain: bool = True,
          work_dir: str = 'res',
          store_name: str = 'project',
          nb_epoch: int = 200,
          batch_size: int = 24,
          resume: bool = False,
          model_path: Optional[str] = None):
    """模型训练。

    Args:
        train_root (str): 训练集存储路径。
        val_root (str): 验证集存储路径。
        model_name (str): 模型名称，可选择的模型名称有：
            - 'resnet50_pmg'
            - 'mobilenetv2_pmg'
        classes_num (int): 分类类别数，默认值：3
        widen_factor (int): 用于控制MobileNet模型复杂度的宽度参数， 默认值：1
        mobilenet_feature_size (int): 用于控制MobileNet模型复杂度的参数， 默认值：512
        pretrain (bool): 是否使用预训练权重，默认值：True
        work_dir (str): 保存训练结果的根目录
        store_name (str): 保存训练结果的文件夹名称，默认值：'project'
        nb_epoch (int): 训练epoch数，默认值：200,
        batch_size (int): batch大小，默认值：24,
        resume (bool): 是否基于之前保存的模型权重进行恢复训练，默认值：False
        model_path (str | None): 用于恢复训练的模型权重的存储路径，默认值：None
    """
    # setup output
    exp_dir = increment_path(work_dir=work_dir, project=store_name)
    try:
        os.stat(exp_dir)
    except Exception:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([transforms.Resize((448, 448)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomRotation(30),
                                          transforms.RandomApply([transforms.ColorJitter(brightness=0.4,
                                                                                         contrast=0.4,
                                                                                         saturation=0.4),],
                                                                 p=0.5),  # 以50%的概率应用高斯模糊
                                          transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.6))], p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.618, 0.506, 0.550), (0.265, 0.292, 0.302)),
                                          ])

    trainset = ImageFolder(root=train_root, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model
    if resume:
        # net = torch.load(model_path)
        net = PMG(resnet50(), 512, 3)
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
    else:
        net = load_model(model_name=model_name,
                         classes_num=classes_num,
                         widen_factor=widen_factor,
                         mobilenet_feature_size=mobilenet_feature_size,
                         pretrain=pretrain,
                         require_grad=True)
    # netp = torch.nn.DataParallel(net, device_ids=[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GPU
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(lr=0.002, params=net.parameters(), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in range(nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            # update learning rate

            # Step 1
            optimizer.zero_grad()
            inputs1 = jigsaw_generator(inputs, 8)  # 图像尺寸缩小8倍，图像分割为8*8个块，块越多，尺寸下采样的程度越小
            output_1, _, _, _ = net(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            _, output_2, _, _ = net(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            _, _, output_3, _ = net(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = net(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 10 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(str(exp_dir) + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                    epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                    train_loss4 / (idx + 1)))

        val_acc, val_acc_com, val_loss = test(net, CELoss, batch_size, val_root)
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com
            # model_name="/model_epoch{}.pt".format(epoch)
            state = {'epoch': epoch,
                     'model': net.state_dict(),
                     'accuracy': val_acc_com,
                     }
            model_name = "/best.pth"
            torch.save(state, str(exp_dir) + model_name)
            with open(str(exp_dir) + '/results_test.txt', 'a') as file:
                file.write('Iteration  %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc, val_acc_com, val_loss))


if __name__ == '__main__':
    train(train_root='/home/wanghan/workspace/datasets/ETI_20250802/train',
          val_root='/home/wanghan/workspace/datasets/ETI_20250802/val',
          model_name='mobilenetv2_pmg',
          classes_num=3,
          widen_factor=1,
          mobilenet_feature_size=320,
          pretrain=True,
          work_dir='res',
          store_name='mobilenet',
          nb_epoch=100,             # number of epoch
          batch_size=24,         # batch size
          resume=False,          # resume training from checkpoint
          model_path='eti/eti_3/best.pth')         # the saved model where you want to resume the training
