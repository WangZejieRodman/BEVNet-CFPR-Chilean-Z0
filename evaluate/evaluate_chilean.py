# evaluate/evaluate_chilean.py
"""
评估Chilean数据集的Place Recognition性能
支持两种模式：
1. 跳过Stage2：使用全局池化（Global Average Pooling）
2. 使用Stage2：使用训练好的AttnVLAD模型
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

from modules.net import AttnVLADHead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pose_data(root, seq):
    """
    加载序列的位置数据（使用z0 CSV）

    Returns:
        dict: {timestamp: (x, y, z)}
    """
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
    """
    加载BEV特征

    Returns:
        features: list of numpy arrays (N, 512, 32, 32)
        timestamps: list of str
        seq_ids: list of str (sequence ID for each feature)
    """
    features = []
    timestamps = []
    seq_ids = []

    for seq in tqdm(seqs, desc="Loading BEV features"):
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
    """
    加载所有特征对应的位置（使用z0 CSV）

    Returns:
        poses: numpy array (N, 2) - x, y坐标
    """
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


def global_pooling_descriptor(bev_features):
    """
    使用全局平均池化生成描述符（跳过Stage2）

    Args:
        bev_features: list of numpy arrays (512, 32, 32)

    Returns:
        descriptors: numpy array (N, 512)
    """
    logger.info("Generating descriptors using Global Average Pooling...")

    descriptors = []
    for fea in tqdm(bev_features, desc="Global pooling"):
        # (512, 32, 32) -> (512,)
        desc = fea.mean(axis=(1, 2))
        descriptors.append(desc)

    descriptors = np.array(descriptors)

    # L2 normalize
    descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

    return descriptors


def vlad_descriptor(bev_features, vlad_model, batch_size=16):
    """
    使用AttnVLAD生成描述符（使用Stage2）

    Args:
        bev_features: list of numpy arrays (512, 32, 32)
        vlad_model: AttnVLADHead模型
        batch_size: batch大小

    Returns:
        descriptors: numpy array (N, D)
    """
    logger.info("Generating descriptors using AttnVLAD...")

    vlad_model.eval()
    descriptors = []

    with torch.no_grad():
        for i in tqdm(range(0, len(bev_features), batch_size), desc="VLAD encoding"):
            batch = bev_features[i:i + batch_size]
            batch_tensor = torch.from_numpy(np.array(batch)).float().to(device)

            # (B, 512, 32, 32) -> (B, D)
            desc = vlad_model(batch_tensor).cpu().numpy()
            descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)

    # L2 normalize
    descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

    return descriptors


def build_ground_truth(query_poses, database_poses, pos_threshold):
    """
    构建Ground Truth：基于GPS距离

    Args:
        query_poses: (M, 2) - query的x,y坐标
        database_poses: (N, 2) - database的x,y坐标
        pos_threshold: 距离阈值（米）

    Returns:
        ground_truth: dict {query_idx: [list of positive database indices]}
    """
    logger.info(f"Building ground truth with threshold {pos_threshold}m...")

    ground_truth = {}

    for i, q_pos in enumerate(tqdm(query_poses, desc="Computing GT")):
        # 计算到所有database的距离
        distances = np.linalg.norm(database_poses - q_pos, axis=1)

        # 找到正样本
        positives = np.where(distances < pos_threshold)[0].tolist()

        ground_truth[i] = positives

    # 统计
    num_with_positives = sum(1 for v in ground_truth.values() if len(v) > 0)
    logger.info(f"Ground truth statistics:")
    logger.info(f"  Total queries: {len(ground_truth)}")
    logger.info(f"  Queries with positives: {num_with_positives}")
    logger.info(f"  Queries without positives: {len(ground_truth) - num_with_positives}")

    return ground_truth


def compute_recall(query_descriptors, database_descriptors, ground_truth, top_k_list):
    """
    计算Recall@K

    Args:
        query_descriptors: (M, D)
        database_descriptors: (N, D)
        ground_truth: dict {query_idx: [positive indices]}
        top_k_list: list of K values

    Returns:
        recalls: dict {k: recall_value}
    """
    logger.info("Computing Recall@K...")

    # 计算相似度矩阵（余弦相似度 = 归一化后的点积）
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
                # 没有正样本的query跳过
                continue

            total += 1

            # 检查top-k中是否有正样本
            top_k_preds = top_indices[query_idx, :k]

            if any(pred in positives for pred in top_k_preds):
                correct += 1

        recall = correct / total if total > 0 else 0.0
        recalls[k] = recall

    return recalls


def evaluate_chilean(config, use_stage2=False):
    """
    主评估函数

    Args:
        config: 配置字典
        use_stage2: 是否使用Stage2模型（True=VLAD, False=全局池化）
    """
    root = config["data_root"]["data_root_folder"]
    eval_config = config["evaluate_config"]

    database_seqs = eval_config["database_seqs"]
    query_seqs = eval_config["query_seqs"]
    pos_threshold = eval_config["pos_threshold"]
    top_k_list = eval_config["top_k"]

    logger.info("=" * 60)
    logger.info("Chilean Dataset Place Recognition Evaluation")
    logger.info("=" * 60)
    logger.info(f"Mode: {'AttnVLAD (Stage2)' if use_stage2 else 'Global Pooling (Skip Stage2)'}")
    logger.info(f"Using Z-compensated data (z0 CSV)")
    logger.info(f"Database sequences: {len(database_seqs)} ({database_seqs[0]}-{database_seqs[-1]})")
    logger.info(f"Query sequences: {len(query_seqs)} ({query_seqs[0]}-{query_seqs[-1]})")
    logger.info(f"Positive threshold: {pos_threshold}m")
    logger.info("=" * 60)

    # 1. 加载Database的BEV特征
    logger.info("\n[1/6] Loading Database BEV features...")
    db_features, db_timestamps, db_seq_ids = load_bev_features(root, database_seqs)
    logger.info(f"Loaded {len(db_features)} database features")

    # 2. 加载Query的BEV特征
    logger.info("\n[2/6] Loading Query BEV features...")
    q_features, q_timestamps, q_seq_ids = load_bev_features(root, query_seqs)
    logger.info(f"Loaded {len(q_features)} query features")

    # 3. 加载位置信息
    logger.info("\n[3/6] Loading pose information...")
    db_poses = load_poses(root, database_seqs, db_timestamps, db_seq_ids)
    q_poses = load_poses(root, query_seqs, q_timestamps, q_seq_ids)
    logger.info(f"Database poses: {db_poses.shape}")
    logger.info(f"Query poses: {q_poses.shape}")

    # 4. 生成全局描述符
    logger.info("\n[4/6] Generating global descriptors...")

    if use_stage2:
        # 使用AttnVLAD
        vlad_model_path = eval_config["test_vlad_model"]

        if not os.path.exists(vlad_model_path):
            logger.error(f"VLAD model not found: {vlad_model_path}")
            logger.error("Please train Stage2 first or use use_stage2=False")
            return

        logger.info(f"Loading VLAD model from {vlad_model_path}")
        vlad_model = AttnVLADHead().to(device)
        checkpoint = torch.load(vlad_model_path)
        vlad_model.load_state_dict(checkpoint['state_dict'])

        db_descriptors = vlad_descriptor(db_features, vlad_model, batch_size=16)
        q_descriptors = vlad_descriptor(q_features, vlad_model, batch_size=16)
    else:
        # 使用全局池化
        db_descriptors = global_pooling_descriptor(db_features)
        q_descriptors = global_pooling_descriptor(q_features)

    logger.info(f"Database descriptors: {db_descriptors.shape}")
    logger.info(f"Query descriptors: {q_descriptors.shape}")

    # 5. 构建Ground Truth
    logger.info("\n[5/6] Building ground truth...")
    ground_truth = build_ground_truth(q_poses, db_poses, pos_threshold)

    # 6. 计算Recall
    logger.info("\n[6/6] Computing Recall@K...")
    recalls = compute_recall(q_descriptors, db_descriptors, ground_truth, top_k_list)

    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mode: {'AttnVLAD (Stage2)' if use_stage2 else 'Global Pooling (Skip Stage2)'}")
    logger.info(f"Using Z-compensated data")
    logger.info(f"Database: {len(db_descriptors)} point clouds from sequences {database_seqs[0]}-{database_seqs[-1]}")
    logger.info(f"Query: {len(q_descriptors)} point clouds from sequences {query_seqs[0]}-{query_seqs[-1]}")
    logger.info(f"Positive threshold: {pos_threshold}m")
    logger.info("-" * 60)

    for k in sorted(recalls.keys()):
        logger.info(f"Recall@{k:2d}: {recalls[k] * 100:6.2f}%")

    logger.info("=" * 60)

    return recalls


if __name__ == '__main__':
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'config/config.yml')
    config = yaml.safe_load(open(config_path))

    # 设置日志
    logger.add("evaluate_chilean_z0.log",
               format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
               encoding='utf-8')

    # 选择评估模式
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Chilean Dataset Place Recognition')
    parser.add_argument('--use_stage2', action='store_true',
                        help='Use Stage2 AttnVLAD model (default: False, use global pooling)')
    args = parser.parse_args()

    # 运行评估
    evaluate_chilean(config, use_stage2=args.use_stage2)