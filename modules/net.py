import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import spconv.pytorch as spconv
import os
import sys
from copy import deepcopy

from modules.netvlad import NetVLADLoupe

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)


def attention(query, key, value):
    """注意力机制"""
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


def MLP(channels: list, do_bn=True):
    """多层感知机"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1],
                            channels[i],
                            kernel_size=1,
                            bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(torch.nn.InstanceNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class MultiHeadedAttention(torch.nn.Module):
    """多头注意力机制"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = torch.nn.ModuleList(
            [deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim,
                                              self.dim * self.num_heads, -1))


class AttentionalPropagation(torch.nn.Module):
    """注意力传播模块"""

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class BottleneckSparse2D(torch.nn.Module):
    """稀疏2D Bottleneck模块（ResNet风格）"""

    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(BottleneckSparse2D, self).__init__()
        self.conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels // 4, 1),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels // 4, kernel_size),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.shotcut_conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.relu = spconv.SparseSequential(torch.nn.ReLU())

    def forward(self, x):
        y = self.conv(x)
        shortcut = self.shotcut_conv(x)
        y = spconv.functional.sparse_add(y, shortcut)
        return self.relu(y)


class UpBlock(torch.nn.Module):
    """上采样块（带skip connection）"""

    def __init__(self, in_channels, shortcut_channels, out_channels) -> None:
        super(UpBlock, self).__init__()
        self.conv1 = BottleneckSparse2D(shortcut_channels, in_channels, 3)
        self.conv2 = BottleneckSparse2D(in_channels, out_channels, 1)

    def forward(self, x, shortcut):
        shortcut = self.conv1(shortcut)
        y = spconv.functional.sparse_add(x, shortcut)
        return self.conv2(y)


class ScoreHead(torch.nn.Module):
    """关键点检测头（局部最大值检测）"""

    def __init__(self, kernel_size=11) -> None:
        super(ScoreHead, self).__init__()
        self.score_pool = torch.nn.AvgPool2d(kernel_size,
                                             stride=1,
                                             padding=kernel_size // 2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size,
                                           stride=1,
                                           padding=kernel_size // 2)
        self.kernel_size = kernel_size

    def forward(self, x, hard_score):
        """
        计算关键点分数（基于局部最大值）
        Args:
            x: 稀疏tensor，特征图
            hard_score: 是否使用硬分数（只保留局部最大值）
        Returns:
            scores: [N, 1] 每个点的关键点分数
        """
        # 归一化特征
        fmax = torch.max(x.features) + 1e-6
        x = x.replace_feature(x.features / fmax)
        x_dense = x.dense()

        # 计算局部平均
        sum_val = self.score_pool(x_dense) * self.kernel_size ** 2
        mask = torch.any(x_dense, dim=1, keepdim=True).float()
        valid_num = self.score_pool(mask) * self.kernel_size ** 2 + 1e-6
        sum_val = sum_val / valid_num

        # 局部最大值分数
        local_max_score = torch.nn.functional.softplus(x_dense - sum_val).permute(0, 2, 3, 1)
        local_max_score = local_max_score[x.indices[:, 0].long(),
        x.indices[:, 1].long(),
        x.indices[:, 2].long()]

        # 通道维度最大值分数
        depth_wise_max = torch.max(x.features, dim=1, keepdims=True)[0]
        depth_wise_max_score = x.features / (1e-6 + depth_wise_max)
        all_scores = local_max_score * depth_wise_max_score
        scores = torch.max(all_scores, dim=1, keepdims=True)[0]

        # 检测是否为局部最大值
        max_temp = self.max_pool(x_dense).permute(0, 2, 3, 1)
        max_temp = max_temp[x.indices[:, 0].long(),
        x.indices[:, 1].long(),
        x.indices[:, 2].long()]
        is_local_max = (x.features == max_temp)
        detected = torch.max(is_local_max.float(), dim=1, keepdims=True)[0]
        scores_hard = scores * detected

        if hard_score:
            scores = scores_hard
        return scores


class Backbone(torch.nn.Module):
    """
    BEV特征提取Backbone
    输入: [B, C, H, W] 体素化的BEV表示
    输出: [B, C', H', W'] 提取的BEV特征
    """

    def __init__(self, inchannels=64) -> None:
        super(Backbone, self).__init__()
        # 下采样路径
        self.dconv_down1 = BottleneckSparse2D(inchannels, inchannels * 2, 11)
        self.dconv_down1_1 = BottleneckSparse2D(inchannels * 2, inchannels * 2, 11)
        self.dconv_down2 = BottleneckSparse2D(inchannels * 2, inchannels * 4, 7)
        self.dconv_down2_1 = BottleneckSparse2D(inchannels * 4, inchannels * 4, 7)
        self.dconv_down3 = BottleneckSparse2D(inchannels * 4, inchannels * 8, 5)
        self.dconv_down3_1 = BottleneckSparse2D(inchannels * 8, inchannels * 8, 5)
        self.dconv_down4 = spconv.SubMConv2d(inchannels * 8,
                                             inchannels * 16,
                                             3,
                                             bias=True)
        # 最大池化
        self.maxpool1 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up1')
        self.maxpool2 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up2')
        self.maxpool3 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up3')

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] dense tensor
        Returns:
            out: [B, C', H', W'] dense tensor
        """
        x = spconv.SparseConvTensor.from_dense(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        x = self.dconv_down1_1(x)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        x = self.dconv_down2_1(x)
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down3_1(x)
        x = self.dconv_down4(x)
        return x.dense()


class FeatureFuse(torch.nn.Module):
    """特征融合模块（使用注意力机制）"""

    def __init__(self, feature_dim, num_heads=1) -> None:
        super(FeatureFuse, self).__init__()
        self.mutihead_attention = AttentionalPropagation(feature_dim, num_heads)

    def forward(self, x, source):
        return (x + self.mutihead_attention(x, source))


class AttnVLADHead(torch.nn.Module):
    """
    CFPR的全局描述符生成头
    使用自注意力增强特征 + NetVLAD聚合
    """

    def __init__(self) -> None:
        super(AttnVLADHead, self).__init__()
        self.self_attention = FeatureFuse(512)
        self.vlad = NetVLADLoupe(
            feature_size=512,  # 固定（来自backbone）
            max_samples=1024,  # 可调：采样点数，越大越慢但更准
            cluster_size=32,  # 可调：聚类中心数，16-64
            output_dim=1024,  # 可调：描述符维度，512-2048
            gating=True,  # 固定：使用门控
            add_batch_norm=True,  # 固定：使用BN
            is_training=True)

    def forward(self, x):
        """
        Args:
            x: [B, 512, 32, 32] BEV特征
        Returns:
            desc: [B, 1024] 全局描述符
        """
        x = x.squeeze(1)
        x = x.reshape(x.shape[0], x.shape[1], -1, 1).squeeze(-1)
        x = self.self_attention(x, x)
        return self.vlad(x.reshape(x.shape[0], x.shape[1], -1, 1))


class OverlapHead(torch.nn.Module):
    """
    重叠度预测头
    输入两个点云的BEV特征，预测它们的重叠度分数
    """

    def __init__(self, inchannels=64) -> None:
        super(OverlapHead, self).__init__()
        self.fusenet16 = FeatureFuse(inchannels * 16)
        self.last_conv16 = spconv.SparseSequential(
            spconv.SubMConv2d(inchannels * 16, inchannels * 8, 3, bias=True),
            torch.nn.BatchNorm1d(inchannels * 8), torch.nn.ReLU(),
            spconv.SubMConv2d(inchannels * 8, 1, 3, bias=True))

    def forward(self, x):
        """
        Args:
            x: [2, C, H, W] 两个点云的特征（batch中第0个和第1个）
        Returns:
            overlap_score: 标量，0-1之间的重叠度分数
            out4: 稀疏tensor，包含每个体素的overlap预测
        """
        x40 = spconv.SparseConvTensor.from_dense(x)
        feature = x40.features
        mask0 = (x40.indices[:, 0] == 0)
        mask1 = (x40.indices[:, 0] == 1)

        # 交叉注意力：点云0关注点云1，点云1关注点云0
        fea1 = self.fusenet16(feature[mask0].permute(1, 0).unsqueeze(0),
                              feature[mask1].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        fea2 = self.fusenet16(feature[mask1].permute(1, 0).unsqueeze(0),
                              feature[mask0].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x40 = x40.replace_feature(torch.cat([fea1, fea2], dim=0))

        # 预测overlap分数（每个体素一个分数）
        out4 = self.last_conv16(x40)
        out4 = out4.replace_feature(torch.sigmoid(out4.features))
        score0 = out4.features[mask0]
        score1 = out4.features[mask1]

        # 平均分数作为整体overlap
        score_sum0 = torch.sum(score0) / len(score0)
        score_sum1 = torch.sum(score1) / len(score1)
        return (score_sum0 + score_sum1) / 2., out4


class BEVNet(torch.nn.Module):
    """
    完整的BEVNet模型（用于Stage1训练）
    包含：
    1. Backbone: 下采样特征提取
    2. 上采样路径: 恢复到原始分辨率
    3. 点级特征头: 描述符、高度权重、关键点分数
    4. OverlapHead: 重叠度预测
    """

    def __init__(self, inchannels=32) -> None:
        super(BEVNet, self).__init__()

        # ============ Backbone部分（下采样） ============
        self.dconv_down1 = BottleneckSparse2D(inchannels, inchannels * 2, 11)
        self.dconv_down1_1 = BottleneckSparse2D(inchannels * 2, inchannels * 2, 11)
        self.dconv_down2 = BottleneckSparse2D(inchannels * 2, inchannels * 4, 7)
        self.dconv_down2_1 = BottleneckSparse2D(inchannels * 4, inchannels * 4, 7)
        self.dconv_down3 = BottleneckSparse2D(inchannels * 4, inchannels * 8, 5)
        self.dconv_down3_1 = BottleneckSparse2D(inchannels * 8, inchannels * 8, 5)
        self.dconv_down4 = spconv.SubMConv2d(inchannels * 8, inchannels * 16, 3, bias=True)

        self.maxpool1 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up1')
        self.maxpool2 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up2')
        self.maxpool3 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up3')

        # ============ 上采样部分（新增！） ============
        self.upsample3 = spconv.SparseInverseConv2d(inchannels * 16,
                                                    inchannels * 8,
                                                    kernel_size=3,
                                                    indice_key="up3")
        self.upsample2 = spconv.SparseInverseConv2d(inchannels * 8,
                                                    inchannels * 4,
                                                    kernel_size=3,
                                                    indice_key="up2")
        self.upsample1 = spconv.SparseInverseConv2d(inchannels * 4,
                                                    inchannels * 2,
                                                    kernel_size=3,
                                                    indice_key="up1")

        self.upblock3 = UpBlock(inchannels * 8, inchannels * 8, inchannels * 8)
        self.upblock2 = UpBlock(inchannels * 4, inchannels * 4, inchannels * 4)
        self.upblock1 = UpBlock(inchannels * 2, inchannels * 2, inchannels * 2)

        # ============ 点级特征头（新增！） ============
        # 生成32维描述符
        self.last_conv = spconv.SubMConv2d(inchannels * 2, 32, 1, bias=True)
        # 生成高度权重（默认64维，对应num_height=64）
        self.weight_conv = spconv.SubMConv2d(inchannels * 2, inchannels, 3, bias=True)
        # 关键点检测头
        self.score_head = ScoreHead(11)

        # ============ OverlapHead部分 ============
        self.fusenet16 = FeatureFuse(inchannels * 16)
        self.last_conv16 = spconv.SparseSequential(
            spconv.SubMConv2d(inchannels * 16, inchannels * 8, 3, bias=True),
            torch.nn.BatchNorm1d(inchannels * 8), torch.nn.ReLU(),
            spconv.SubMConv2d(inchannels * 8, 1, 3, bias=True))

    def extract_feature(self, x):
        """只提取BEV特征（用于特征提取，不需要上采样）"""
        x = spconv.SparseConvTensor.from_dense(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        x = self.dconv_down1_1(x)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        x = self.dconv_down2_1(x)
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down3_1(x)
        x = self.dconv_down4(x)
        return x.dense()

    def calc_overlap(self, x40):
        """只计算重叠度（用于评估）"""
        x40 = spconv.SparseConvTensor.from_dense(x40)
        feature = x40.features
        mask0 = (x40.indices[:, 0] == 0)
        mask1 = (x40.indices[:, 0] == 1)
        fea1 = self.fusenet16(feature[mask0].permute(1, 0).unsqueeze(0),
                              feature[mask1].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        fea2 = self.fusenet16(feature[mask1].permute(1, 0).unsqueeze(0),
                              feature[mask0].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x40 = x40.replace_feature(torch.cat([fea1, fea2], dim=0))
        out4 = self.last_conv16(x40)
        out4 = out4.replace_feature(torch.sigmoid(out4.features))
        score0 = out4.features[mask0]
        score1 = out4.features[mask1]
        score_sum0 = torch.sum(score0) / len(score0)
        score_sum1 = torch.sum(score1) / len(score1)
        return (score_sum0 + score_sum1) / 2.

    def forward(self, x, eval=False):
        """
        完整前向传播（用于训练）
        Args:
            x: [2, C, H, W] 两个点云的体素化表示
            eval: 是否为评估模式
        Returns:
            out: 稀疏tensor，原始分辨率的点级特征 [N, 32+num_height+1]
            out4: 稀疏tensor，overlap预测结果
            x4: dense tensor，下采样后的backbone特征
        """
        x = spconv.SparseConvTensor.from_dense(x)

        # ============ 下采样路径 ============
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        x = self.dconv_down1_1(x)

        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        x = self.dconv_down2_1(x)

        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down3_1(x)

        x = self.dconv_down4(x)

        # 保存低分辨率特征（用于overlap预测）
        x4 = x
        x40 = x

        # ============ Overlap预测分支（在低分辨率） ============
        feature = x40.features
        mask0 = (x40.indices[:, 0] == 0)
        mask1 = (x40.indices[:, 0] == 1)

        # 交叉注意力融合
        fea1 = self.fusenet16(feature[mask0].permute(1, 0).unsqueeze(0),
                              feature[mask1].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        fea2 = self.fusenet16(feature[mask1].permute(1, 0).unsqueeze(0),
                              feature[mask0].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x40 = x40.replace_feature(torch.cat([fea1, fea2], dim=0))

        # Overlap预测
        out4 = self.last_conv16(x40)
        out4 = out4.replace_feature(torch.sigmoid(out4.features))

        # ============ 上采样路径（新增！用于点级特征） ============
        x = self.upsample3(x)
        x = self.upblock3(x, conv3)

        x = self.upsample2(x)
        x = self.upblock2(x, conv2)

        x = self.upsample1(x)
        x_last = self.upblock1(x, conv1)

        # ============ 点级特征生成（新增！） ============
        # 1. 32维描述符
        x = self.last_conv(x_last)

        # 2. 高度权重（num_height维，这里是64维）
        weight = self.weight_conv(x_last)
        weight_feature = weight.features
        weight_feature = torch.sigmoid(weight_feature)  # 归一化到[0,1]
        weight = weight.replace_feature(weight_feature)

        # 3. 关键点分数（1维）
        scores = self.score_head(x, eval)

        # ============ 拼接所有特征 ============
        # 归一化描述符
        features = torch.nn.functional.normalize(x.features, dim=1)

        # 如果在eval模式，用overlap分数加权关键点分数
        if eval:
            div = [x.spatial_shape[i] // out4.spatial_shape[i]
                   for i in range(len(x.spatial_shape))]
            indices = x.indices.clone()
            indices[:, 1] = indices[:, 1] // div[0]
            indices[:, 2] = indices[:, 2] // div[1]
            overlap_score = out4.dense().squeeze(1)[indices[:, 0].long(),
            indices[:, 1].long(),
            indices[:, 2].long()]
            scores = scores * overlap_score.unsqueeze(-1)

        # 拼接: [32维描述符 + num_height维高度权重 + 1维分数]
        x = x.replace_feature(
            torch.cat([features, weight_feature, scores.reshape(-1, 1)], dim=1)
        )
        # x.features: [N, 32+64+1=97]

        return x, out4, x4


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Testing BEVNet with upsampling...")
    print("=" * 60)

    # 测试完整BEVNet
    model = BEVNet(32).to(device)
    x = torch.randn(2, 32, 256, 256).to(device)

    print(f"\nInput shape: {x.shape}")
    out, out4, x4 = model(x)

    print(f"\n✓ Forward pass successful!")
    print(f"  out: {out.spatial_shape}, features shape: {out.features.shape}")
    print(f"  out4: {out4.spatial_shape}, features shape: {out4.features.shape}")
    print(f"  x4: {x4.shape}")

    # 检查特征维度
    expected_dim = 32 + 64 + 1  # 描述符 + 高度权重 + 分数
    assert out.features.shape[1] == expected_dim, \
        f"Feature dimension mismatch! Expected {expected_dim}, got {out.features.shape[1]}"
    print(f"\n✓ Feature dimension correct: {out.features.shape[1]} = 32 + 64 + 1")

    # 检查分辨率
    assert out.spatial_shape == [256, 256], \
        f"Spatial resolution mismatch! Expected [256, 256], got {out.spatial_shape}"
    print(f"✓ Spatial resolution correct: {out.spatial_shape}")

    print("\n" + "=" * 60)
    print("All tests PASSED! ✓")
    print("=" * 60)