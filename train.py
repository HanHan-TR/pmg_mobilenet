import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.fileio import increment_path, yaml_load, yaml_save
from core.dataset import creat_dataset
from utils import jigsaw_generator, load_model, cosine_anneal_schedule, test


def train(opt):
    """模型训练。
    """
    # setup output
    exp_dir = increment_path(work_dir=opt.work_dir, project=opt.project, name=opt.name)

    try:
        os.stat(exp_dir)
    except Exception:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()

    # network_hyp, dataset_cfg, train_settings = yaml_load(opt.network_hyp), yaml_load(opt.dataset_cfg), yaml_load(opt.train_settings)
    network_hyp = yaml_load(opt.network_hyp)
    dataset_cfg = yaml_load(opt.dataset_cfg)
    train_settings = yaml_load(opt.train_settings)
    yaml_save(exp_dir / 'network_hyp.yaml', network_hyp)
    yaml_save(exp_dir / 'dataset_cfg.yaml', dataset_cfg)
    yaml_save(exp_dir / 'train_settings.yaml', train_settings)

    # Data
    print('==> Preparing data..')  # 根据是否进行交叉验证使用不同的创建方式进行数据集创建
    train_dataset = creat_dataset(cfg=dataset_cfg, mode='train')
    val_dataset = creat_dataset(cfg=dataset_cfg, mode='val')
    trainloader = DataLoader(train_dataset,
                             batch_size=train_settings['batch_size_train'],
                             shuffle=True,
                             num_workers=4)
    valloader = DataLoader(val_dataset,
                           batch_size=train_settings['batch_size_val'],
                           shuffle=False,
                           num_workers=4)
    # Model
    net = load_model(model_name=network_hyp['model_name'],
                     classes_num=dataset_cfg['classes_num'],
                     widen_factor=network_hyp['widen_factor'],
                     mobilenet_feature_size=network_hyp['mobilenet_feature_size'],
                     pretrain=network_hyp['pretrain'],
                     require_grad=network_hyp['require_grad'])
    if opt.resume:
        checkpoint = torch.load(opt.load_from)
        net.load_state_dict(checkpoint['model'])

    # netp = torch.nn.DataParallel(net, device_ids=[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GPU
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(lr=0.002, params=net.parameters(), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD([{'params': net.classifier_concat.parameters(), 'lr': 0.002},
                           {'params': net.conv_block1.parameters(), 'lr': 0.002},
                           {'params': net.classifier1.parameters(), 'lr': 0.002},
                           {'params': net.conv_block2.parameters(), 'lr': 0.002},
                           {'params': net.classifier2.parameters(), 'lr': 0.002},
                           {'params': net.conv_block3.parameters(), 'lr': 0.002},
                           {'params': net.classifier3.parameters(), 'lr': 0.002},
                           {'params': net.backbone.parameters(), 'lr': 0.0002}],
                          momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0
    # lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lr = train_settings['lr']

    for epoch in range(train_settings['nb_epoch']):
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
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch,
                                                                       train_settings['nb_epoch'], lr[nlr])

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < train_settings['batch_size_train']:
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

        val_acc, val_acc_com, val_loss = test(net=net,
                                              dataloader=valloader,
                                              criterion=CELoss,
                                              device=device)
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
                file.write('Iteration  %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (epoch, val_acc, val_acc_com, val_loss))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    # Step 1 ------------------- 设置配置文件 ------------------------------------------------------------------------------
    parser.add_argument('--network-hyp',
                        default=ROOT / 'hyp/pmg_mobilenetv2_w1_fs320.yaml', help='network config file path')
    parser.add_argument('--train-settings',
                        default=ROOT / 'hyp/train_settings.yaml', help='dataset config file path')
    parser.add_argument('--dataset-cfg',
                        default=ROOT / 'hyp/dataset_lma_fine-tuning.yaml', help='dataset config file path')
    # Step 2 ------------------- 设置训练结果保存路径 ------------------------------------------------------------------------------
    parser.add_argument('--work-dir',
                        default=ROOT / 'res/train', help='the dir to save logs and models')
    parser.add_argument('--project', type=str,
                        default='mobilenet', help='save to work-dir/project')
    parser.add_argument('--name', default='exp', help='save to work-dir/project/name')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume',
                        action='store_true',
                        default=False,
                        help='resume from the latest checkpoint automatically.')
    parser.add_argument('--load-from', type=str,
                        default='',
                        help='the checkpoint file to load weights from')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == '__main__':
    opt = parse_args()
    train(opt=opt)         # the saved model where you want to resume the training
