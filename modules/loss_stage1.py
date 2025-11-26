import torch
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_fscore_support
import random
import sys
import os

# 导入cdist函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.utils import utils


def pair_loss(features_x,
              features_y,
              T_x,
              T_y,
              points0=None,
              points1=None,
              points_xy0=None,
              points_xy1=None,
              num_height=32,
              min_z=-4,
              height=7,
              search_radiu=0.3):
    """
    点对损失函数：计算匹配点对之间的特征描述符loss + 关键点检测loss

    Args:
        features_x: 点云0的特征 [N0, C] (C=32+num_height+1: descriptor+weights+score)
        features_y: 点云1的特征 [N1, C]
        T_x: 点云0的位姿变换矩阵 [4, 4]
        T_y: 点云1的位姿变换矩阵 [4, 4]
        points0: 点云0的原始3D坐标 [N0, 3]
        points1: 点云1的原始3D坐标 [N1, 3]
        points_xy0: 点云0在BEV网格中的xy坐标 [N0, 2]
        points_xy1: 点云1在BEV网格中的xy坐标 [N1, 2]
        num_height: 高度预测的bin数量
        min_z: 最小高度
        height: 高度范围
        search_radiu: 匹配点搜索半径

    Returns:
        total_loss: 总损失
        desc_loss: 描述符损失
        det_loss: 检测损失
        score_loss: 分数损失
        z_loss: 高度预测损失
        z_loss0, z_loss1: 与原始点云的高度对齐损失
        acc: 匹配准确率
    """
    # 分离特征的不同部分
    weights_x = features_x[:, 32:32 + num_height]  # 高度权重
    weights_y = features_y[:, 32:32 + num_height]
    scores_x = features_x[:, 32 + num_height]  # 关键点分数
    scores_y = features_y[:, 32 + num_height]
    features_x = features_x[:, :32]  # 描述符
    features_y = features_y[:, :32]

    points_xy0 = points_xy0.to(device=features_x.device)
    points_xy1 = points_xy1.to(device=features_x.device)

    # 从高度权重预测z坐标
    dz = height / num_height
    pose_x = points_xy0[:, :2].float()
    pose_y = points_xy1[:, :2].float()
    z_id = torch.arange(0, num_height).float().to(features_x.device).reshape(
        1, num_height) * dz + min_z + dz / 2.

    pos_xz = torch.sum(weights_x * z_id, dim=-1, keepdim=True)
    pos_yz = torch.sum(weights_y * z_id, dim=-1, keepdim=True)

    # 构建3D坐标 [x, y, z, 1]
    pose_x = torch.cat([pose_x, pos_xz, torch.ones_like(pose_x[:, 0]).view(-1, 1)], dim=1)
    pose_y = torch.cat([pose_y, pos_yz, torch.ones_like(pose_y[:, 0]).view(-1, 1)], dim=1)

    # 应用位姿变换到世界坐标系
    pose_x = T_x.to(device=features_x.device) @ pose_x.T
    pose_x = pose_x.T[:, :3]
    pose_y = T_y.to(device=features_x.device) @ pose_y.T
    pose_y = pose_y.T[:, :3]
    tempx = pose_x.detach().cpu().numpy()
    tempy = pose_y.detach().cpu().numpy()

    # 使用KDTree找到匹配点对
    tree = KDTree(tempy[:, :2])
    ind_nn = tree.query_radius(tempx[:, :2], r=search_radiu)
    matchs = []
    for i in range(len(ind_nn)):
        if len(ind_nn[i]) > 0:
            random.shuffle(ind_nn[i])
            matchs.append([i, ind_nn[i][0]])  # 只取最近的一个匹配

    if len(matchs) < 1024:
        # 匹配点太少，返回None
        return None, None, None, None, None, None, None, None

    matchs = np.array(matchs, dtype='int32')

    # 高度预测损失
    z_loss = torch.nn.functional.l1_loss(pose_x[matchs[:, 0], 2],
                                         pose_y[matchs[:, 1], 2])

    # 随机采样1024个匹配对
    selected_ind = np.random.choice(range(len(matchs)), 1024, replace=False)
    matchs = matchs[selected_ind]
    score_x = scores_x[matchs[:, 0]]
    score_y = scores_y[matchs[:, 1]]
    match_x = torch.from_numpy(tempx[matchs[:, 0], :2]).to(features_x.device)
    match_y = torch.from_numpy(tempy[matchs[:, 1], :2]).to(features_x.device)
    features_x_selected = features_x[matchs[:, 0]]
    features_y_selected = features_y[matchs[:, 1]]

    # 计算描述符loss（circle loss）和检测loss
    desc_loss, acc, _, _, _, dist = circleloss(features_x_selected,
                                               features_y_selected, match_x,
                                               match_y)
    det_loss = detloss(dist, score_x, score_y)
    score_loss = torch.nn.functional.l1_loss(score_x, score_y)

    # 计算预测高度与原始点云高度的对齐损失
    x_selected = pose_x[matchs[:, 0], :3]
    y_selected = pose_y[matchs[:, 1], :3]

    points0 = points0.detach().numpy()
    points1 = points1.detach().numpy()
    pcd0 = utils.make_open3d_point_cloud(points0)
    pcd0.transform(T_x.detach().numpy())
    pcd1 = utils.make_open3d_point_cloud(points1)
    pcd1.transform(T_y.detach().numpy())
    points0 = np.asarray(pcd0.points)
    points1 = np.asarray(pcd1.points)

    # 对点云0
    tree_x = KDTree(points0)
    ind_nn = tree_x.query(x_selected.detach().cpu().numpy(), 1)[1]
    match_pair = []
    for i in range(len(ind_nn)):
        match_pair.append([i, ind_nn[i][0]])
    match_pair = np.asarray(match_pair, dtype='int32')
    matched_z0 = points0[match_pair[:, 1], 2]
    z_loss0 = torch.nn.functional.l1_loss(
        x_selected[:, 2],
        torch.from_numpy(matched_z0).to(features_x.device).float())

    # 对点云1
    tree_y = KDTree(points1)
    ind_nn = tree_y.query(y_selected.detach().cpu().numpy(), 1)[1]
    match_pair = []
    for i in range(len(ind_nn)):
        match_pair.append([i, ind_nn[i][0]])
    match_pair = np.asarray(match_pair, dtype='int32')
    matched_z1 = points1[match_pair[:, 1], 2]
    z_loss1 = torch.nn.functional.l1_loss(
        y_selected[:, 2],
        torch.from_numpy(matched_z1).to(features_x.device).float())

    return desc_loss + det_loss + z_loss + z_loss0 + z_loss1, desc_loss, det_loss, score_loss, z_loss, z_loss0, z_loss1, acc


def circleloss(anchor,
               positive,
               anchor_keypts,
               positive_keypts,
               dist_type='euclidean',
               log_scale=10,
               safe_radius=1.0,
               pos_margin=0.1,
               neg_margin=1.4):
    """
    Circle Loss用于度量学习

    Args:
        anchor: anchor特征 [N, D]
        positive: positive特征 [N, D]
        anchor_keypts: anchor关键点坐标 [N, 2]
        positive_keypts: positive关键点坐标 [N, 2]
        dist_type: 距离度量方式
        log_scale: log-sum-exp的缩放因子
        safe_radius: 负样本安全半径（超过此距离才算负样本）
        pos_margin: 正样本margin
        neg_margin: 负样本margin

    Returns:
        loss: circle loss
        accuracy: 匹配准确率（furthest_positive < closest_negative的比例）
        furthest_positive: 最远的正样本距离
        average_negative: 平均负样本距离
        0: 占位符
        dists: 距离矩阵
    """
    # 计算特征距离矩阵
    dists = utils.cdist(anchor, positive, metric=dist_type)
    # 计算关键点空间距离矩阵
    dist_keypts = utils.cdist(anchor_keypts, positive_keypts, metric='euclidean')

    # 正样本mask: 对角线元素（对应的anchor-positive对）
    pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
    pos_mask = torch.eq(torch.unsqueeze(pids, dim=1),
                        torch.unsqueeze(pids, dim=0))
    # 负样本mask: 空间距离大于safe_radius的点对
    neg_mask = dist_keypts > safe_radius

    # 找到最远的正样本和最近的负样本
    furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
    closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)
    average_negative = (torch.sum(dists, dim=-1) - furthest_positive) / (dists.shape[0] - 1)
    diff = furthest_positive - closest_negative
    accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]

    # Circle loss计算
    pos = dists - 1e5 * neg_mask.float()
    pos_weight = (pos - pos_margin).detach()
    pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)
    lse_positive_row = torch.logsumexp(log_scale * (pos - pos_margin) * pos_weight, dim=-1)
    lse_positive_col = torch.logsumexp(log_scale * (pos - pos_margin) * pos_weight, dim=-2)

    neg = dists + 1e5 * (~neg_mask).float()
    neg_weight = (neg_margin - neg).detach()
    neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)
    lse_negative_row = torch.logsumexp(log_scale * (neg_margin - neg) * neg_weight, dim=-1)
    lse_negative_col = torch.logsumexp(log_scale * (neg_margin - neg) * neg_weight, dim=-2)

    loss_col = torch.nn.functional.softplus(lse_positive_row + lse_negative_row) / log_scale
    loss_row = torch.nn.functional.softplus(lse_positive_col + lse_negative_col) / log_scale
    loss = loss_col + loss_row

    return torch.mean(loss), accuracy, furthest_positive.tolist(), average_negative.tolist(), 0, dists


def detloss(dists, anc_score, pos_score):
    """
    检测loss：鼓励关键点的分数与匹配难度相关

    Args:
        dists: 距离矩阵 [N, N]
        anc_score: anchor关键点分数 [N]
        pos_score: positive关键点分数 [N]

    Returns:
        loss: 检测loss
    """
    pids = torch.FloatTensor(np.arange(len(anc_score))).to(anc_score.device)
    pos_mask = torch.eq(torch.unsqueeze(pids, dim=1),
                        torch.unsqueeze(pids, dim=0))
    furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
    closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)
    # 分数越高，应该使positive更近、negative更远
    loss = (furthest_positive - closest_negative) * (anc_score + pos_score).squeeze(-1)
    return torch.mean(loss)


def get_weighted_bce_loss(prediction, gt):
    """
    加权的二元交叉熵损失（用于处理类别不平衡）

    Args:
        prediction: 预测值 [N]
        gt: 真值标签 [N] (0或1)

    Returns:
        w_class_loss: 加权BCE loss
        cls_precision: 精确率
        cls_recall: 召回率
    """
    loss = torch.nn.BCELoss(reduction='none')
    class_loss = loss(prediction, gt)

    # 计算类别权重
    weights = torch.ones_like(gt)
    w_negative = gt.sum() / gt.size(0)
    w_positive = 1 - w_negative
    w_negative = max(w_negative, 0.1)
    w_positive = max(w_positive, 0.1)

    weights[gt >= 0.5] = w_positive
    weights[gt < 0.5] = w_negative
    w_class_loss = torch.mean(weights * class_loss)

    # 计算精确率和召回率
    predicted_labels = prediction.detach().cpu().round().numpy()
    cls_precision, cls_recall, _, _ = precision_recall_fscore_support(
        gt.cpu().numpy(), predicted_labels, average='binary')

    return w_class_loss, cls_precision, cls_recall


def overlap_loss(out, T_x, T_y, min_x=-50, max_x=50, show=False):
    """
    重叠区域预测损失
    预测两个点云中哪些区域是重叠的

    Args:
        out: 稀疏卷积输出 (SparseTensor)，包含两个点云的特征和overlap分数
        T_x: 点云0的位姿 [4, 4]
        T_y: 点云1的位姿 [4, 4]
        min_x, max_x: BEV范围
        show: 是否可视化

    Returns:
        w_class_loss: 加权BCE loss
        cls_precision: 精确率
        cls_recall: 召回率
    """
    # 分离两个点云的特征
    mask0 = (out.indices[:, 0] == 0)
    mask1 = (out.indices[:, 0] == 1)
    indi0 = out.indices[mask0, 1:]
    indi1 = out.indices[mask1, 1:]
    score0 = torch.clamp(out.features[mask0].squeeze(-1), min=0, max=1)
    score1 = torch.clamp(out.features[mask1].squeeze(-1), min=0, max=1)

    # 将BEV网格索引转换为实际坐标
    dx = (max_x - min_x) / out.spatial_shape[0]
    pose_x = indi0 * dx + min_x + dx / 2.
    pose_y = indi1 * dx + min_x + dx / 2.

    pose_x = torch.cat([
        pose_x,
        torch.zeros_like(pose_x[:, 0]).view(-1, 1),
        torch.ones_like(pose_x[:, 0]).view(-1, 1)
    ], dim=1)
    pose_y = torch.cat([
        pose_y,
        torch.zeros_like(pose_y[:, 0]).view(-1, 1),
        torch.ones_like(pose_y[:, 0]).view(-1, 1)
    ], dim=1)

    # 转换到世界坐标系
    pose_x = T_x.to(device=score0.device) @ pose_x.T
    pose_y = T_y.to(device=score0.device) @ pose_y.T
    pose_x = pose_x.T[:, :3]
    pose_y = pose_y.T[:, :3]

    tempx = pose_x.detach().cpu().numpy()
    tempy = pose_y.detach().cpu().numpy()

    # 基于空间距离生成真值标签
    # 对点云0: 找距离点云1最近的点
    tree_y = KDTree(tempy[:, :2])
    ind_nn = tree_y.query(tempx[:, :2], 1)[0].reshape(-1)
    pos_id0 = []
    neg_id0 = []
    for i in range(len(ind_nn)):
        if ind_nn[i] < dx:  # 在一个体素范围内认为是重叠区域
            pos_id0.append(i)
        else:
            neg_id0.append(i)

    # 对点云1
    tree_x = KDTree(tempx[:, :2])
    ind_nn = tree_x.query(tempy[:, :2], 1)[0].reshape(-1)
    pos_id1 = []
    neg_id1 = []
    for i in range(len(ind_nn)):
        if ind_nn[i] < dx:
            pos_id1.append(i)
        else:
            neg_id1.append(i)

    # 构建真值标签
    gt0 = torch.zeros_like(score0).to(score0.device)
    gt0[pos_id0] = 1
    gt1 = torch.zeros_like(score1).to(score0.device)
    gt1[pos_id1] = 1

    # 计算加权BCE loss
    w_class_loss, cls_precision, cls_recall = get_weighted_bce_loss(
        torch.cat([score0, score1]), torch.cat([gt0, gt1]))

    return w_class_loss, cls_precision, cls_recall


def dist_loss(out, T_x, T_y, min_x=-50, max_x=50):
    """
    特征距离loss：鼓励匹配点的特征相似

    Args:
        out: 稀疏卷积输出，包含点云特征
        T_x, T_y: 位姿变换
        min_x, max_x: BEV范围

    Returns:
        desc_loss: 描述符loss
        acc: 匹配准确率
    """
    mask0 = (out.indices[:, 0] == 0)
    mask1 = (out.indices[:, 0] == 1)
    indi0 = out.indices[mask0, 1:]
    indi1 = out.indices[mask1, 1:]
    fea0 = torch.nn.functional.normalize(out.features[mask0], dim=1)
    fea1 = torch.nn.functional.normalize(out.features[mask1], dim=1)

    # 转换到世界坐标
    dx = (max_x - min_x) / out.spatial_shape[0]
    pose_x = indi0 * dx + min_x + dx / 2.
    pose_y = indi1 * dx + min_x + dx / 2.

    pose_x = torch.cat([
        pose_x,
        torch.zeros_like(pose_x[:, 0]).view(-1, 1),
        torch.ones_like(pose_x[:, 0]).view(-1, 1)
    ], dim=1)
    pose_y = torch.cat([
        pose_y,
        torch.zeros_like(pose_y[:, 0]).view(-1, 1),
        torch.ones_like(pose_y[:, 0]).view(-1, 1)
    ], dim=1)
    pose_x = T_x.to(device=fea0.device) @ pose_x.T
    pose_y = T_y.to(device=fea0.device) @ pose_y.T
    pose_x = pose_x.T[:, :3]
    pose_y = pose_y.T[:, :3]

    tempx = pose_x.detach().cpu().numpy()
    tempy = pose_y.detach().cpu().numpy()

    # 找匹配点对
    tree = KDTree(tempy[:, :2])
    ind_nn = tree.query_radius(tempx[:, :2], r=dx)
    matchs = []
    for i in range(len(ind_nn)):
        if len(ind_nn[i]) > 0:
            random.shuffle(ind_nn[i])
            matchs.append([i, ind_nn[i][0]])
    if len(matchs) < 4:
        return None, None

    matchs = np.array(matchs, dtype='int32')
    if len(matchs) > 512:
        selected_ind = np.random.choice(range(len(matchs)), 512, replace=False)
        matchs = matchs[selected_ind]

    features_x_selected = fea0[matchs[:, 0]]
    features_y_selected = fea1[matchs[:, 1]]
    match_x = torch.from_numpy(tempx[matchs[:, 0], :2]).to(fea0.device)
    match_y = torch.from_numpy(tempy[matchs[:, 1], :2]).to(fea0.device)

    # 计算circle loss
    desc_loss, acc, _, _, _, _ = circleloss(features_x_selected,
                                            features_y_selected,
                                            match_x,
                                            match_y,
                                            safe_radius=3 * dx,
                                            dist_type='euclidean')
    return desc_loss, acc