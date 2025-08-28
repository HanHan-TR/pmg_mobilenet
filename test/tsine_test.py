import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def prepare_features(feature_tensors):
    """
    将多个形状为 [1, 1024, 1, 1] 的张量转换为二维矩阵
    参数:
        feature_tensors: 包含多个张量的列表，每个张量形状为 [1, 1024, 1, 1]
    返回:
        features_matrix: 形状为 (n_samples, 1024) 的 NumPy 数组
    """
    # 移除冗余维度并拼接
    flattened = [tensor.squeeze() for tensor in feature_tensors]  # 移除维度1和3
    stacked = torch.stack(flattened)                             # 堆叠为 [n, 1024]
    return stacked.cpu().numpy()                                  # 转为NumPy


def tsne_visualization(features, labels=None, perplexity=30, n_iter=1000):
    """
    执行 t-SNE 降维并可视化
    参数:
        features: 形状为 (n_samples, n_features) 的NumPy数组
        labels: 样本标签（用于着色），若无则传入None
        perplexity: t-SNE 困惑度参数
        n_iter: 迭代次数
    """
    # 数据标准化（避免量纲影响距离计算）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 打乱数据（避免顺序影响）
    if labels is not None:
        features_scaled, labels_shuffled = shuffle(features_scaled, labels)
    else:
        features_scaled = shuffle(features_scaled)

    # 初始化 t-SNE 模型
    tsne = TSNE(n_components=2,          # 降维到2D
                perplexity=perplexity,    # 控制邻域大小（通常5~50）
                n_iter=n_iter,            # 最小建议500~1000
                random_state=42,          # 固定随机种子
                verbose=1                 # 显示进度
                )

    # 执行降维
    tsne_results = tsne.fit_transform(features_scaled)

    # 可视化
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=labels_shuffled,    # 按标签着色
            cmap='tab10',         # 调色板
            alpha=0.7
        )
        plt.colorbar(scatter, label='Class Labels')
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)

    plt.title('t-SNE Visualization of Feature Vectors')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(alpha=0.3)
    plt.savefig('test.png')


# 示例使用 -------------------------------------------------
if __name__ == "__main__":
    # 1. 模拟数据：生成10个示例向量（形状 [1,1024,1,1]）
    feature_list = [torch.randn(1, 1024, 1, 1) for _ in range(10)]

    # 2. 可选：生成模拟标签（如无标签则设为None）
    example_labels = np.random.randint(0, 3, size=10)  # 3个类别

    # 3. 数据预处理
    features_matrix = prepare_features(feature_list)

    # 4. 执行 t-SNE 可视化
    tsne_visualization(
        features_matrix,
        labels=example_labels,    # 传入None则不显示标签
        perplexity=3,            # 小数据集建议调低
        n_iter=500
    )
