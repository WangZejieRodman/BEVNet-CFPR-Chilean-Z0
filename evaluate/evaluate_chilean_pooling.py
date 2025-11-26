# evaluate/evaluate_chilean_pooling.py
"""
评估Chilean数据集的Place Recognition性能 - 支持多种池化方法
对比不同池化方法的效果，评估Stage1 BEV特征的质量
使用z0 CSV文件
"""
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

# 导入池化方法
from pooling_methods import (
    GlobalAvgPoolingDescriptor,
    MaxPoolingDescriptor,
    GeMPoolingDescriptor,
    MixedPoolingDescriptor
)

# 导入AttnVLAD (用于对比)
from modules.net import AttnVLADHead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pose_data(root, seq):
    """加载序列的位置数据（使用z0 CSV）"""
    csv_path = os.path.join(root, seq, "pointcloud_pos_ori_20m_10overlap_z0.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"Pose file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    pose_dict = {}
    for _, row in df.iterrows():
        timestamp = str(int(row['timestamp']))
        x, y, z = row['x'], row['y'], row['z']
        pose_dict[timestamp] = (x, y, z)

    return pose_dict


def load_bev_features(root, seqs):
    """加载BEV特征"""
    features = []
    timestamps = []
    seq_ids = []

    for seq in tqdm(seqs, desc="加载BEV特征"):
        fea_folder = os.path.join(root, seq, "BEV_FEA")

        if not os.path.exists(fea_folder):
            logger.warning(f"BEV feature folder not found: {fea_folder}")
            continue

        fea_files = sorted([f for f in os.listdir(fea_folder) if f.endswith('.npy')])

        for fea_file in fea_files:
            fea_path = os.path.join(fea_folder, fea_file)
            fea = np.load(fea_path)  # (512, 32, 32)

            timestamp = fea_file.replace('.npy', '')

            features.append(fea)
            timestamps.append(timestamp)
            seq_ids.append(seq)

    return features, timestamps, seq_ids


def load_poses(root, seqs, timestamps, seq_ids):
    """加载所有特征对应的位置"""
    poses = []

    # 按序列加载位置数据
    seq_pose_dict = {}
    for seq in seqs:
        seq_pose_dict[seq] = load_pose_data(root, seq)

    # 为每个特征找到对应位置
    for timestamp, seq in zip(timestamps, seq_ids):
        if seq in seq_pose_dict and timestamp in seq_pose_dict[seq]:
            x, y, z = seq_pose_dict[seq][timestamp]
            poses.append([x, y])
        else:
            logger.warning(f"Pose not found for {seq}/{timestamp}, using (0, 0)")
            poses.append([0.0, 0.0])

    return np.array(poses)


def vlad_descriptor(bev_features, vlad_model, batch_size=16):
    """使用AttnVLAD生成描述符"""
    logger.info("使用AttnVLAD生成描述符...")

    vlad_model.eval()
    descriptors = []

    with torch.no_grad():
        for i in tqdm(range(0, len(bev_features), batch_size), desc="VLAD编码"):
            batch = bev_features[i:i + batch_size]
            batch_tensor = torch.from_numpy(np.array(batch)).float().to(device)

            # (B, 512, 32, 32) -> (B, 1024)
            desc = vlad_model(batch_tensor).cpu().numpy()
            descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)

    # L2归一化
    descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

    return descriptors


def build_ground_truth(query_poses, database_poses, pos_threshold):
    """构建Ground Truth：基于GPS距离"""
    logger.info(f"构建ground truth，阈值={pos_threshold}m...")

    ground_truth = {}

    for i, q_pos in enumerate(tqdm(query_poses, desc="计算GT")):
        # 计算到所有database的距离
        distances = np.linalg.norm(database_poses - q_pos, axis=1)

        # 找到正样本
        positives = np.where(distances < pos_threshold)[0].tolist()

        ground_truth[i] = positives

    # 统计
    num_with_positives = sum(1 for v in ground_truth.values() if len(v) > 0)
    logger.info(f"Ground truth统计:")
    logger.info(f"  总查询数: {len(ground_truth)}")
    logger.info(f"  有正样本的查询: {num_with_positives}")
    logger.info(f"  无正样本的查询: {len(ground_truth) - num_with_positives}")

    return ground_truth


def compute_recall(query_descriptors, database_descriptors, ground_truth, top_k_list):
    """计算Recall@K"""
    logger.info("计算Recall@K...")

    # 计算相似度矩阵（余弦相似度）
    similarity = query_descriptors @ database_descriptors.T  # (M, N)

    # 对每个query，找到最相似的top_k个database
    max_k = max(top_k_list)
    top_indices = np.argsort(-similarity, axis=1)[:, :max_k]  # (M, max_k)

    # 计算各个K值的recall
    recalls = {}

    for k in top_k_list:
        correct = 0
        total = 0

        for query_idx, positives in ground_truth.items():
            if len(positives) == 0:
                continue

            total += 1

            # 检查top-k中是否有正样本
            top_k_preds = top_indices[query_idx, :k]

            if any(pred in positives for pred in top_k_preds):
                correct += 1

        recall = correct / total if total > 0 else 0.0
        recalls[k] = recall

    return recalls


def evaluate_with_pooling_method(config, pooling_method):
    """
    使用指定的池化方法进行评估

    Args:
        config: 配置字典
        pooling_method: 池化方法对象或'attnvlad'
    """
    root = config["data_root"]["data_root_folder"]
    eval_config = config["evaluate_config"]

    database_seqs = eval_config["database_seqs"]
    query_seqs = eval_config["query_seqs"]
    pos_threshold = eval_config["pos_threshold"]
    top_k_list = eval_config["top_k"]

    # 确定方法名称
    if pooling_method == 'attnvlad':
        method_name = "AttnVLAD (Stage2 Trained)"
    else:
        method_name = pooling_method.name

    logger.info("=" * 60)
    logger.info(f"评估方法: {method_name}")
    logger.info("=" * 60)
    logger.info(f"使用Z补偿数据 (z0 CSV)")
    logger.info(f"Database序列: {len(database_seqs)} ({database_seqs[0]}-{database_seqs[-1]})")
    logger.info(f"Query序列: {len(query_seqs)} ({query_seqs[0]}-{query_seqs[-1]})")
    logger.info(f"正样本阈值: {pos_threshold}m")
    logger.info("=" * 60)

    # 1. 加载Database的BEV特征
    logger.info("\n[1/6] 加载Database BEV特征...")
    db_features, db_timestamps, db_seq_ids = load_bev_features(root, database_seqs)
    logger.info(f"已加载 {len(db_features)} 个database特征")

    # 2. 加载Query的BEV特征
    logger.info("\n[2/6] 加载Query BEV特征...")
    q_features, q_timestamps, q_seq_ids = load_bev_features(root, query_seqs)
    logger.info(f"已加载 {len(q_features)} 个query特征")

    # 3. 加载位置信息
    logger.info("\n[3/6] 加载位置信息...")
    db_poses = load_poses(root, database_seqs, db_timestamps, db_seq_ids)
    q_poses = load_poses(root, query_seqs, q_timestamps, q_seq_ids)
    logger.info(f"Database位置: {db_poses.shape}")
    logger.info(f"Query位置: {q_poses.shape}")

    # 4. 生成全局描述符
    logger.info("\n[4/6] 生成全局描述符...")

    if pooling_method == 'attnvlad':
        # 使用训练好的AttnVLAD
        vlad_model_path = eval_config["test_vlad_model"]

        if not os.path.exists(vlad_model_path):
            logger.error(f"VLAD模型未找到: {vlad_model_path}")
            return None

        logger.info(f"加载VLAD模型: {vlad_model_path}")
        vlad_model = AttnVLADHead().to(device)
        checkpoint = torch.load(vlad_model_path)
        vlad_model.load_state_dict(checkpoint['state_dict'])

        db_descriptors = vlad_descriptor(db_features, vlad_model, batch_size=16)
        q_descriptors = vlad_descriptor(q_features, vlad_model, batch_size=16)
    else:
        # 使用池化方法
        db_descriptors = pooling_method.generate(db_features)
        q_descriptors = pooling_method.generate(q_features)

    logger.info(f"Database描述符: {db_descriptors.shape}")
    logger.info(f"Query描述符: {q_descriptors.shape}")

    # 5. 构建Ground Truth
    logger.info("\n[5/6] 构建Ground Truth...")
    ground_truth = build_ground_truth(q_poses, db_poses, pos_threshold)

    # 6. 计算Recall
    logger.info("\n[6/6] 计算Recall@K...")
    recalls = compute_recall(q_descriptors, db_descriptors, ground_truth, top_k_list)

    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info(f"评估结果: {method_name}")
    logger.info("=" * 60)
    logger.info(f"使用Z补偿数据")
    logger.info(f"Database: {len(db_descriptors)} 点云")
    logger.info(f"Query: {len(q_descriptors)} 点云")
    logger.info(f"描述符维度: {db_descriptors.shape[1]}")
    logger.info("-" * 60)

    for k in sorted(recalls.keys()):
        logger.info(f"Recall@{k:2d}: {recalls[k] * 100:6.2f}%")

    logger.info("=" * 60)

    return recalls


def run_comparison(config):
    """运行所有池化方法的对比实验"""

    # 定义要测试的方法
    methods = {
        'Global Avg': GlobalAvgPoolingDescriptor(),
        'Max': MaxPoolingDescriptor(),
        'GeM (p=3)': GeMPoolingDescriptor(p=3.0),
        'GeM (p=4)': GeMPoolingDescriptor(p=4.0),
        'Mixed': MixedPoolingDescriptor(),
        'AttnVLAD': 'attnvlad'  # 特殊标记
    }

    # 存储所有结果
    all_results = {}

    # 逐个测试
    for method_name, method in methods.items():
        logger.info(f"\n\n{'=' * 80}")
        logger.info(f"开始测试: {method_name}")
        logger.info(f"{'=' * 80}\n")

        try:
            recalls = evaluate_with_pooling_method(config, method)
            all_results[method_name] = recalls
        except Exception as e:
            logger.error(f"方法 {method_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[method_name] = None
            continue

    # 打印对比表格
    logger.info("\n\n" + "=" * 80)
    logger.info("所有方法对比结果 (使用Z补偿数据)")
    logger.info("=" * 80)

    # 表头
    k_values = sorted(list(all_results.values())[0].keys())
    header = f"{'方法':<20} " + " ".join([f"R@{k:<4}" for k in k_values])
    logger.info(header)
    logger.info("-" * 80)

    # 每个方法的结果
    for method_name, recalls in all_results.items():
        if recalls is None:
            continue
        row = f"{method_name:<20} "
        row += " ".join([f"{recalls[k] * 100:5.2f}%" for k in k_values])
        logger.info(row)

    logger.info("=" * 80)

    # 分析
    logger.info("\n关键发现:")

    # 比较Global Avg和Max
    if 'Global Avg' in all_results and 'Max' in all_results:
        avg_r1 = all_results['Global Avg'][1]
        max_r1 = all_results['Max'][1]
        diff = (max_r1 - avg_r1) * 100
        logger.info(f"1. Max vs Global Avg (R@1): {diff:+.2f}%")
        if diff > 5:
            logger.info("   → Max明显优于Avg，说明BEV特征有区分度，峰值特征重要")
        elif diff < -5:
            logger.info("   → Avg优于Max，说明全局统计信息更重要")
        else:
            logger.info("   → 差异不大，说明聚合方式影响有限")

    # 比较GeM和其他
    if 'GeM (p=3)' in all_results:
        gem_r1 = all_results['GeM (p=3)'][1]
        logger.info(f"2. GeM (p=3) R@1: {gem_r1 * 100:.2f}%")
        if 'Global Avg' in all_results:
            if gem_r1 > all_results['Global Avg'][1]:
                logger.info("   → GeM优于平均池化")

    # 比较AttnVLAD和最佳池化方法
    if 'AttnVLAD' in all_results and all_results['AttnVLAD'] is not None:
        vlad_r1 = all_results['AttnVLAD'][1]
        pooling_results = {k: v[1] for k, v in all_results.items() if k != 'AttnVLAD'}
        best_pooling = max(pooling_results.values())
        best_pooling_name = [k for k, v in pooling_results.items() if v == best_pooling][0]

        logger.info(f"3. AttnVLAD vs 最佳池化方法:")
        logger.info(f"   AttnVLAD: {vlad_r1 * 100:.2f}%")
        logger.info(f"   最佳池化 ({best_pooling_name}): {best_pooling * 100:.2f}%")
        logger.info(f"   Stage2增益: {(vlad_r1 - best_pooling) * 100:+.2f}%")

        if vlad_r1 - best_pooling > 0.1:  # 10%以上提升
            logger.info("   → Stage2训练带来显著提升，Stage2是关键")
        elif vlad_r1 - best_pooling < 0.05:  # 5%以内
            logger.info("   → Stage2提升有限，Stage1 BEV特征已经足够好")
        else:
            logger.info("   → Stage2有一定作用，但不是主要因素")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    # 配置路径
    config_path = '/home/wzj/pan1/BEVNet-CFPR/config/config.yml'

    # 加载配置
    config = yaml.safe_load(open(config_path))

    # 设置日志
    logger.add("/home/wzj/pan1/BEVNet-CFPR/outputs/evaluate_pooling_comparison_z0.log",
               format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
               encoding='utf-8')

    # 运行对比实验
    run_comparison(config)