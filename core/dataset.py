from torchvision import transforms
from torchvision.datasets import ImageFolder

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))


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

transform_test = transforms.Compose([transforms.Resize((448, 448)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.618, 0.506, 0.550), (0.265, 0.292, 0.302))
                                     ])


def creat_dataset(cfg: dict,
                  mode: str = 'train'):
    assert mode in ('train', 'val'), f"The dataset mode must be one of 'train', 'val', but got {mode}"
    data_dir = str(Path(cfg['path']) / cfg[mode])

    if mode == 'val':
        dataset = ImageFolder(root=data_dir, transform=transform_test)
    elif mode == 'train':
        dataset = ImageFolder(root=data_dir, transform=transform_train)
    else:
        dataset = ImageFolder(root=data_dir, transform=None)

    # 手动设置类别映射关系
    if cfg['custom_mapping'] is not None:
        dataset.class_to_idx = cfg['custom_mapping']
        dataset.classes = list(cfg['custom_mapping'].keys())  # 更新类别列表
    return dataset
