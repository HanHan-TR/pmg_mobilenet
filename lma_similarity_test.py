import cv2 as cv
from tqdm import tqdm
import os.path as osp
import albumentations as A
import torch
from torchvision import transforms

import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from models.mobilenet_v2 import MobileNetV2
from core.initialize import load_state_dict
from core.fileio import extract_number
from tools.feature_similarity import cosine_distance, dot_product_distance, euclidean_distance

TQDM_BAR_FORMAT = '{l_bar}{bar:20}{r_bar}'

data_root = Path('../datasets/LMA_standard')
img_names = [
    'LMA-1a_0006831.jpg', 'LMA-2_0002115.jpg', 'LMA-3_0002607.jpg', 'LMA-4_0002001.jpg', 'LMA-5_0001500.jpg',
    'LMA-6_0000900.jpg', 'LMA-7_0000800.jpg', 'LMA-8_0000500.jpg', 'LMA-11_0023800.jpg', 'LMA-13a_0012252.jpg',
    'LMA-14a_0002400.jpg', 'LMA-15_0001800.jpg ', 'LMA-17_0001800.jpg', 'LMA-18_0007500.jpg ', 'LMA-19_0002500.jpg ',
    'LMA-20a_0006000.jpg', 'LMA-21_0007700.jpg ', 'LMA-22_0025400.jpg ', 'LMA-23a_0006500.jpg', 'LMA-24a_0002900.jpg',
    'LMA-25_0002452.jpg ', 'LMA-27_0008476.jpg ', 'LMA-29b_0005456.jpg', 'LMA-30b_0003152.jpg', 'LMA-31_0024640.jpg ', 'LMA-32a_0007752.jpg'
]
video_names = [
    'LMA-1a.mp4',  # 'LMA-1b.mp4 ',
    'LMA-2.mp4', 'LMA-3.mp4', 'LMA-4.mp4', 'LMA-5.mp4',

    'LMA-6.mp4', 'LMA-7.mp4', 'LMA-8.mp4', 'LMA-11.mp4', 'LMA-13a.mp4',  # 'LMA-13b.mp4',

    'LMA-14a.mp4',  # 'LMA-14b.mp4',
    'LMA-15.mp4', 'LMA-17.mp4', 'LMA-18.mp4', 'LMA-19.mp4',

    'LMA-20a.mp4',  # 'LMA-20b.mp4',
    'LMA-21.mp4', 'LMA-22.mp4', 'LMA-23a.mp4',  # 'LMA-23b.mp4'， 'LMA-23c.mp4',
    'LMA-24a.mp4',  # 'LMA-24b.mp4',

    'LMA-25.mp4', 'LMA-27.mp4', 'LMA-29b.mp4',  # 'LMA-29a.mp4','LMA-29c.mp4', 'LMA-29e.mp4','LMA-29d.mp4'
    'LMA-30b.mp4',  # 'LMA-30c.mp4',
    'LMA-31.mp4', 'LMA-32a.mp4',  # 'LMA-32b.mp4', 'LMA-32d.mp4',
]

# transform = transforms.Compose([transforms.Resize((224, 224)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.618, 0.506, 0.550), (0.265, 0.292, 0.302))
#                                 ])

transform = A.Compose([  # 归一化与转换成 Tensor 模式
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.618, 0.506, 0.550), std=(0.265, 0.292, 0.302)),  # !! mean 与 std 的数值需要根据不同的数据集改变
    # Convert image and mask to PyTorch tensors
    A.ToTensorV2(),  # !! 请使用最新版的albumentations （2.0.6版本是OK的）
])


def feat_diff(model,
              video,
              image,
              data_root=Path('../datasets/LMA_standard')):
    # 初始化 dist_summary 字典，并直接添加距离键
    dist_summary = {
        'video': None,  # 这些键通过 update 添加，但这里预先定义
        'standard_image': None,
        'standard_frame_idx': None,
        'dist_cosine': {'before': {'min': None, 'max': None, 'mean': None}, 'after': {'min': None, 'max': None, 'mean': None}},  # 初始化嵌套结构
        'dist_dot': {'before': {'min': None, 'max': None, 'mean': None}, 'after': {'min': None, 'max': None, 'mean': None}},
        'dist_euc': {'before': {'min': None, 'max': None, 'mean': None}, 'after': {'min': None, 'max': None, 'mean': None}}
    }
    model.eval().to('cuda')
    standard_frame_indx = extract_number(image)  # 标准图像在第几帧
    dist_summary.update(video=video, standard_image=image, standard_frame_indx=standard_frame_indx)

    # 提取标准图像的特征表示
    standard_img_path = str(data_root / image)
    standard_image = cv.imread(standard_img_path, cv.COLOR_BGR2RGB)
    standard_input = transform(image=standard_image)['image'].unsqueeze(0).to('cuda')
    standard_feat = model(standard_input)[0]

    video = str(data_root / video)
    cap = cv.VideoCapture(video)
    num_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    process_bar = tqdm(range(num_frame), total=num_frame, bar_format=TQDM_BAR_FORMAT)

    dist_cosine, dist_dot, dist_euc = dict(before=[], after=[]), dict(before=[], after=[]), dict(before=[], after=[])

    for count in process_bar:
        # 读取视频文件
        ret, frame = cap.read()
        if not ret:
            break

        # 提取当前帧图像的特征表示
        frame_input = transform(image=frame)['image'].unsqueeze(0).to('cuda')
        frame_feat = model(frame_input)[0]
        d_cosine = cosine_distance(standard_feat, frame_feat)
        d_dot = dot_product_distance(standard_feat, frame_feat)
        d_euc = euclidean_distance(standard_feat, frame_feat)

        # 计算特征的差异性

        if count > standard_frame_indx:
            dist_cosine['after'].append(d_cosine)
            dist_dot['after'].append(d_dot)
            dist_euc['after'].append(d_euc)
            process_bar.desc = f"Video {dist_summary['video']} mean distance (after): cosine: {sum(dist_cosine['after'])/ len(dist_cosine['after'])}, " \
                               f"dot: {sum(dist_dot['after'])/ len(dist_dot['after'])}, " \
                               f"euclidean: {sum(dist_euc['after'])/ len(dist_euc['after'])}, "
        else:
            dist_cosine['before'].append(d_cosine)
            dist_dot['before'].append(d_dot)
            dist_euc['before'].append(d_euc)

            process_bar.desc = f"Video {dist_summary['video']} mean distance (before): cosine: {sum(dist_cosine['before'])/ len(dist_cosine['before'])}, " \
                               f"dot: {sum(dist_dot['before'])/ len(dist_dot['before'])}, " \
                               f"euclidean: {sum(dist_euc['before'])/ len(dist_euc['before'])}, "

    title = f"\n ========= Video: {dist_summary['video']}, standard image: {dist_summary['standard_image']} =============== \n"
    with open('lma_similarity_test.log', 'a') as file:
        file.write(title)
    print(title)
    for i in ('dist_cosine', 'dist_dot', 'dist_euc'):
        dist_summary[i]['before']['min'] = min(eval(i)['before'])
        dist_summary[i]['before']['max'] = max(eval(i)['before'])
        dist_summary[i]['before']['mean'] = sum(eval(i)['before']) / len(eval(i)['before'])

        dist_summary[i]['after']['min'] = min(eval(i)['after'])
        dist_summary[i]['after']['max'] = max(eval(i)['after'])
        dist_summary[i]['after']['mean'] = sum(eval(i)['after']) / len(eval(i)['after'])

        summary = f"-------------> {i}: \n" \
                  f"[before]: \n min: {dist_summary[i]['before']['min']:.3f}, mean: {dist_summary[i]['before']['mean']:.3f}, max: {dist_summary[i]['before']['max']:.3f} \n" \
                  f"[after]: \n min: {dist_summary[i]['after']['min']:.3f}, mean: {dist_summary[i]['after']['mean']:.3f}, max: {dist_summary[i]['after']['max']:.3f} \n"

        with open('lma_similarity_test.log', 'a') as file:
            file.write(summary)
        print(summary)


if __name__ == "__main__":
    # 构建特征提取网络
    model = MobileNetV2(out_indices=(8,), widen_factor=1.0)
    # 加载预训练权重
    filename = 'pretrained_ckpt/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, prefix='backbone.')

    # 设置不需要对参数求梯度，即冻结所有参数
    for params in model.parameters():
        params.requires_grad = False

    assert len(img_names) == len(video_names)

    for i in range(len(img_names)):
        image_name = img_names[i]
        video_name = video_names[i]
        feat_diff(model=model, video=video_name, image=image_name)


diff_before, diff_after = [], []
