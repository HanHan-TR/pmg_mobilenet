import yaml
from typing import Union
import re
import sys
import os
from pathlib import Path, PosixPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))


def check_suffix(file='hyp.yaml', suffix=('.yaml',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'


def check_file(file, suffix=''):
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file


def check_yaml(file, suffix=('.yaml', '.yml')):
    # checking suffix
    return check_file(file, suffix)


def yaml_load(file='data.yaml'):
    file = check_yaml(file)
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


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


def extract_number(filename):
    """
    从文件名中提取数字部分并转换为整数
    :param filename: 图像文件名（可包含路径）
    :return: 提取的整数值，若无数字则返回None
    """
    # 获取纯文件名（去除路径和扩展名）
    basename = os.path.basename(filename)
    base_without_ext = os.path.splitext(basename)[0]

    # 提取所有连续数字序列
    numbers = re.findall(r'\d+', base_without_ext)

    if not numbers:
        return None  # 无数字序列时返回None

    # 取最后一个数字序列（通常位于文件名末尾）
    last_number_str = numbers[-1]
    return int(last_number_str)  # 字符串转整数
