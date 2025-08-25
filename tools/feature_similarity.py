import torch
import torch.nn.functional as F


# 方法1：余弦距离（值越小越相似）
def cosine_distance(v1, v2):
    """计算余弦距离（范围[0,1]，越小越相似）"""
    v1_flat = v1.squeeze()
    v2_flat = v2.squeeze()
    # 计算余弦相似度并转换为距离
    sim = F.cosine_similarity(v1_flat.unsqueeze(0), v2_flat.unsqueeze(0), dim=1)
    return (1 - sim).item()  # 越小越相似[5,7](@ref)

# 方法2：点积距离（值越小越相似）


def dot_product_distance(v1, v2):
    """计算归一化点积距离（范围[0,1]，越小越相似）"""
    v1_flat = v1.view(-1)
    v2_flat = v2.view(-1)
    # 归一化向量后计算点积
    norm_v1 = F.normalize(v1_flat, dim=0)
    norm_v2 = F.normalize(v2_flat, dim=0)
    dot = torch.dot(norm_v1, norm_v2)
    return (1 - dot).item()  # 越小越相似[5](@ref)

# 方法3：归一化欧氏距离（值越小越相似）


def euclidean_distance(v1, v2):
    """计算归一化欧氏距离（范围[0,1)，越小越相似）"""
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    # 计算原始欧氏距离并归一化
    dist = torch.sqrt(torch.sum((v1_flat - v2_flat) ** 2))
    return (dist / (1 + dist)).item()
