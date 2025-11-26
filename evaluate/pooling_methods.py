# evaluate/pooling_methods.py
"""
各种池化方法用于BEV特征聚合
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class GlobalAvgPoolingDescriptor:
    """全局平均池化"""

    def __init__(self):
        self.name = "Global Average Pooling"

    def generate(self, bev_features, batch_size=16, device='cuda:0'):
        """
        Args:
            bev_features: list of numpy arrays (32, 256, 256)
            batch_size: batch大小
            device: 设备
        Returns:
            descriptors: numpy array (N, 32)
        """
        print(f"使用 {self.name} 生成描述符...")

        descriptors = []
        for fea in tqdm(bev_features, desc="Global Avg Pooling"):
            # (32, 256, 256) -> (32,)
            desc = fea.mean(axis=(1, 2))
            descriptors.append(desc)

        descriptors = np.array(descriptors)

        # L2归一化
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

        return descriptors


class MaxPoolingDescriptor:
    """全局最大池化"""

    def __init__(self):
        self.name = "Global Max Pooling"

    def generate(self, bev_features, batch_size=16, device='cuda:0'):
        """
        Args:
            bev_features: list of numpy arrays (32, 256, 256)
            batch_size: batch大小
            device: 设备
        Returns:
            descriptors: numpy array (N, 32)
        """
        print(f"使用 {self.name} 生成描述符...")

        descriptors = []
        for fea in tqdm(bev_features, desc="Global Max Pooling"):
            # (32, 256, 256) -> (32,)
            desc = fea.max(axis=(1, 2))
            descriptors.append(desc)

        descriptors = np.array(descriptors)

        # L2归一化
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

        return descriptors


class GeMPoolingDescriptor:
    """Generalized Mean Pooling"""

    def __init__(self, p=3.0):
        """
        Args:
            p: GeM参数，p=1时等于平均池化，p=inf时等于最大池化
               一般取3-4之间
        """
        self.name = f"GeM Pooling (p={p})"
        self.p = p

    def generate(self, bev_features, batch_size=16, device='cuda:0'):
        """
        Args:
            bev_features: list of numpy arrays (32, 256, 256)
            batch_size: batch大小
            device: 设备
        Returns:
            descriptors: numpy array (N, 32)
        """
        print(f"使用 {self.name} 生成描述符...")

        descriptors = []
        for fea in tqdm(bev_features, desc=f"GeM Pooling (p={self.p})"):
            # GeM: (1/N * sum(x^p))^(1/p)
            # 加入epsilon避免数值问题
            eps = 1e-6
            fea_clamped = np.maximum(fea, eps)  # 避免负值或0

            # (32, 256, 256) -> (32,)
            desc = np.power(
                np.mean(np.power(fea_clamped, self.p), axis=(1, 2)),
                1.0 / self.p
            )
            descriptors.append(desc)

        descriptors = np.array(descriptors)

        # L2归一化
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

        return descriptors


class MixedPoolingDescriptor:
    """混合池化：拼接平均池化和最大池化"""

    def __init__(self):
        self.name = "Mixed Pooling (Avg + Max)"

    def generate(self, bev_features, batch_size=16, device='cuda:0'):
        """
        Args:
            bev_features: list of numpy arrays (32, 256, 256)
            batch_size: batch大小
            device: 设备
        Returns:
            descriptors: numpy array (N, 64) - 拼接平均和最大
        """
        print(f"使用 {self.name} 生成描述符...")

        descriptors = []
        for fea in tqdm(bev_features, desc="Mixed Pooling"):
            # (32, 256, 256) -> (32,) + (32,) = (64,)
            desc_avg = fea.mean(axis=(1, 2))
            desc_max = fea.max(axis=(1, 2))
            desc = np.concatenate([desc_avg, desc_max])
            descriptors.append(desc)

        descriptors = np.array(descriptors)

        # L2归一化
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

        return descriptors


if __name__ == '__main__':
    # 测试代码
    print("测试各种池化方法...")

    # 生成随机BEV特征
    np.random.seed(42)
    test_features = [np.random.randn(32, 256, 256).astype(np.float32) for _ in range(10)]

    # 测试各种池化
    methods = [
        GlobalAvgPoolingDescriptor(),
        MaxPoolingDescriptor(),
        GeMPoolingDescriptor(p=3.0),
        MixedPoolingDescriptor()
    ]

    for method in methods:
        desc = method.generate(test_features, batch_size=4)
        print(f"{method.name}: {desc.shape}")
        print(f"  范数检查: {np.linalg.norm(desc[0]):.4f} (应该接近1.0)")
        print()