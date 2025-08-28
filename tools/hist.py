import matplotlib.pyplot as plt
import numpy as np

# 假设这是您的三个列表数据（取值范围0-1）
list1 = np.random.rand(1000)  # 随机生成1000个0-1之间的数
list2 = np.random.beta(2, 5, 1000)  # 偏左分布
list3 = np.random.beta(5, 2, 1000)  # 偏右分布


def plot_hist(datas: dict):
    # 创建图形和坐标轴
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)  # 20个区间，区间宽度0.05
    for key, value in datas.items():
        if isinstance(value, dict) and value.get('data'):
            dist_list = value.get('data')
            weights = np.ones_like(dist_list) / len(dist_list)
            plt.hist(dist_list,
                     bins=bins,
                     weights=weights * 100.0,
                     alpha=0.4,
                     color='red',
                     label=key)

    # 添加图表元素
    plt.title('Distribution Comparison of Three Lists', fontsize=14)
    plt.xlabel('Value Range (0-1)', fontsize=12)
    plt.ylabel('Percentage of Elements (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)

    # 显示图表
    plt.tight_layout()
    plt.show()
