# evaluate/evaluate_feature_quality_cross_seq.py
"""
跨序列评估Chilean数据集Stage1 BEV特征的质量
Database: 160-189
Query: 190-209
避免时序连续性干扰，真实评估特征区分能力
使用z0 CSV文件
"""
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def load_bev_features_and_poses(root, seqs):
    """加载BEV特征和GPS位置（使用z0 CSV）"""
    features = []
    poses = []
    seq_labels = []

    for seq in tqdm(seqs, desc="加载数据"):
        fea_folder = os.path.join(root, seq, "BEV_FEA")

        if not os.path.exists(fea_folder):
            logger.warning(f"BEV特征文件夹不存在: {fea_folder}")
            continue

        pose_file = os.path.join(root, seq, "pointcloud_pos_ori_20m_10overlap_z0.csv")

        if not os.path.exists(pose_file):
            logger.warning(f"位置文件不存在: {pose_file}")
            continue

        # 读取CSV：timestamp, x, y, z
        df = pd.read_csv(pose_file, sep=',')

        # 提取xy坐标用于计算距离
        pose_xy = df[['x', 'y']].values.astype(np.float32)
        pose_dict = {}
        for i, (_, row) in enumerate(df.iterrows()):
            timestamp = str(int(row['timestamp']))
            pose_dict[timestamp] = tuple(pose_xy[i])

        # 读取特征
        fea_files = sorted([f for f in os.listdir(fea_folder) if f.endswith('.npy')])

        for fea_file in fea_files:
            timestamp = fea_file.replace('.npy', '')

            if timestamp not in pose_dict:
                continue

            fea_path = os.path.join(fea_folder, fea_file)
            fea = np.load(fea_path)  # (512, 32, 32)

            features.append(fea)
            poses.append(pose_dict[timestamp])
            seq_labels.append(seq)

    return features, poses, seq_labels


def pooling_features(features, method='avg'):
    """对BEV特征进行池化"""
    descriptors = []

    for fea in features:
        if method == 'avg':
            desc = fea.mean(axis=(1, 2))
        elif method == 'max':
            desc = fea.max(axis=(1, 2))
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        # L2归一化
        desc = desc / (np.linalg.norm(desc) + 1e-8)
        descriptors.append(desc)

    return np.array(descriptors)


def compute_distance_matrix(desc1, desc2):
    """计算两组描述符之间的欧氏距离矩阵"""
    # 使用余弦相似度转欧氏距离
    similarity = desc1 @ desc2.T
    distances = np.sqrt(2 - 2 * similarity)
    return distances


def metric1_intra_inter_distance(db_features, db_poses, q_features, q_poses,
                                 pos_threshold=15.0, neg_threshold=30.0, pooling='avg'):
    """指标1：类内/类间距离比"""
    print(f"\n[指标1] 计算类内/类间距离比 (pooling={pooling})...")

    # 池化
    db_desc = pooling_features(db_features, pooling)
    q_desc = pooling_features(q_features, pooling)

    # 计算所有query到database的距离
    all_distances = compute_distance_matrix(q_desc, db_desc)  # (Q, D)

    # 构建正负样本对
    intra_distances = []
    inter_distances = []

    for i, q_pos in enumerate(tqdm(q_poses, desc="计算类内/类间距离")):
        for j, db_pos in enumerate(db_poses):
            gps_dist = np.linalg.norm(np.array(q_pos) - np.array(db_pos))
            fea_dist = all_distances[i, j]

            if gps_dist < pos_threshold:
                intra_distances.append(fea_dist)
            elif gps_dist > neg_threshold:
                inter_distances.append(fea_dist)

    # 随机采样（避免太多）
    if len(intra_distances) > 5000:
        intra_distances = np.random.choice(intra_distances, 5000, replace=False)
    if len(inter_distances) > 5000:
        inter_distances = np.random.choice(inter_distances, 5000, replace=False)

    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)

    intra_mean = intra_distances.mean()
    inter_mean = inter_distances.mean()
    ratio = inter_mean / (intra_mean + 1e-8)

    print(f"  类内距离: {intra_mean:.4f} (共{len(intra_distances)}对)")
    print(f"  类间距离: {inter_mean:.4f} (共{len(inter_distances)}对)")
    print(f"  区分度比: {ratio:.4f}")

    return {
        'intra_mean': intra_mean,
        'inter_mean': inter_mean,
        'ratio': ratio,
        'intra_distances': intra_distances,
        'inter_distances': inter_distances
    }


def metric2_nn_accuracy(db_features, db_poses, q_features, q_poses,
                        k_list=[1, 5, 10, 25], pos_threshold=15.0, pooling='avg'):
    """指标2：最近邻准确率"""
    print(f"\n[指标2] 计算最近邻准确率 (pooling={pooling})...")

    # 池化
    db_desc = pooling_features(db_features, pooling)
    q_desc = pooling_features(q_features, pooling)

    # 计算相似度矩阵
    similarity = q_desc @ db_desc.T  # (Q, D)

    # 对每个query找最近邻
    accuracies = {}

    for k in k_list:
        correct_count = 0
        total_count = 0

        for i, q_pos in enumerate(tqdm(q_poses, desc=f"NN@{k}", leave=False)):
            # 找最相似的K个
            top_k_indices = np.argsort(-similarity[i])[:k]

            # 计算GPS距离
            q_pos_arr = np.array(q_pos)
            db_poses_arr = np.array([db_poses[j] for j in top_k_indices])
            distances = np.linalg.norm(db_poses_arr - q_pos_arr, axis=1)

            # 统计正样本数量
            num_correct = np.sum(distances < pos_threshold)
            correct_count += num_correct
            total_count += k

        accuracy = correct_count / total_count if total_count > 0 else 0
        accuracies[k] = accuracy
        print(f"  NN Accuracy@{k}: {accuracy * 100:.2f}%")

    return accuracies


def metric3_clustering_quality(db_features, q_features, num_clusters=50, pooling='avg'):
    """指标3：聚类质量"""
    print(f"\n[指标3] 计算聚类质量 (pooling={pooling})...")

    # 池化
    db_desc = pooling_features(db_features, pooling)
    q_desc = pooling_features(q_features, pooling)

    # 合并
    all_desc = np.vstack([db_desc, q_desc])

    # 采样（避免太慢）
    if len(all_desc) > 2000:
        idx = np.random.choice(len(all_desc), 2000, replace=False)
        all_desc_sampled = all_desc[idx]
    else:
        all_desc_sampled = all_desc

    # K-means聚类
    print(f"  K-means聚类 (K={num_clusters})...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_desc_sampled)

    # Silhouette Score
    silhouette = silhouette_score(all_desc_sampled, labels, sample_size=1000)

    print(f"  Silhouette Score: {silhouette:.4f}")
    if silhouette > 0.5:
        print(f"  评价: 优秀 (>0.5)")
    elif silhouette > 0.3:
        print(f"  评价: 良好 (0.3-0.5)")
    elif silhouette > 0.2:
        print(f"  评价: 一般 (0.2-0.3)")
    else:
        print(f"  评价: 较差 (<0.2)")

    return silhouette


def metric4_distance_distribution(intra_distances, inter_distances, pooling='avg'):
    """指标4：绘制正负样本距离分布图"""
    print(f"\n[指标4] 绘制距离分布图 (pooling={pooling})...")

    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, max(intra_distances.max(), inter_distances.max()), 50)

    plt.hist(intra_distances, bins=bins, alpha=0.6,
             label='Positive Pairs (Intra-class)',
             color='blue', density=True)
    plt.hist(inter_distances, bins=bins, alpha=0.6,
             label='Negative Pairs (Inter-class)',
             color='red', density=True)

    plt.axvline(intra_distances.mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Intra-class Mean: {intra_distances.mean():.3f}')
    plt.axvline(inter_distances.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Inter-class Mean: {inter_distances.mean():.3f}')

    plt.xlabel('Feature Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('BEV Feature Distance Distribution (Z-compensated)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    save_path = f'/home/wzj/pan1/BEVNet-CFPR/outputs/distance_distribution_{pooling}_z0.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图表已保存: {save_path}")
    plt.close()

    # 计算重叠度
    hist1, _ = np.histogram(intra_distances, bins=100, density=True)
    hist2, _ = np.histogram(inter_distances, bins=100, density=True)
    overlap = np.sum(np.minimum(hist1, hist2)) / 100

    print(f"  分布重叠度: {overlap * 100:.2f}%")

    return overlap


def metric5_linear_separability(db_features, db_poses, q_features, q_poses,
                                pos_threshold=15.0, neg_threshold=30.0, pooling='avg'):
    """指标5：线性可分性"""
    print(f"\n[指标5] 计算线性可分性 (pooling={pooling})...")

    # 池化
    db_desc = pooling_features(db_features, pooling)
    q_desc = pooling_features(q_features, pooling)

    # 构建正负样本
    X_pos = []
    X_neg = []

    print("  构建正负样本数据集...")
    for i, q_pos in enumerate(tqdm(q_poses, desc="构建样本", leave=False)):
        for j, db_pos in enumerate(db_poses):
            gps_dist = np.linalg.norm(np.array(q_pos) - np.array(db_pos))

            if gps_dist < pos_threshold:
                diff = q_desc[i] - db_desc[j]
                X_pos.append(diff)
            elif gps_dist > neg_threshold:
                diff = q_desc[i] - db_desc[j]
                X_neg.append(diff)

        # 限制样本数量
        if len(X_pos) > 1000 and len(X_neg) > 1000:
            break

    # 采样到相同数量
    sample_size = min(1000, len(X_pos), len(X_neg))
    if len(X_pos) > sample_size:
        X_pos = [X_pos[i] for i in np.random.choice(len(X_pos), sample_size, replace=False)]
    if len(X_neg) > sample_size:
        X_neg = [X_neg[i] for i in np.random.choice(len(X_neg), sample_size, replace=False)]

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    print(f"  正样本: {len(X_pos)}, 负样本: {len(X_neg)}")

    # 训练测试划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练线性SVM
    print("  训练线性SVM...")
    svm = LinearSVC(random_state=42, max_iter=5000)
    svm.fit(X_train, y_train)

    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)

    print(f"  训练集准确率: {train_acc * 100:.2f}%")
    print(f"  测试集准确率: {test_acc * 100:.2f}%")

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    }


def evaluate_cross_sequence(config):
    """跨序列评估主函数"""
    root = config["data_root"]["data_root_folder"]

    # 数据库和查询序列
    database_seqs = [str(i) for i in range(160, 190)]  # 160-189
    query_seqs = [str(i) for i in range(190, 210)]  # 190-209

    # 参数
    pos_threshold = 15.0
    neg_threshold = 30.0

    logger.info("=" * 80)
    logger.info("跨序列BEV特征质量评估 (使用Z补偿数据)")
    logger.info("=" * 80)
    logger.info(f"数据根目录: {root}")
    logger.info(f"Database序列: {len(database_seqs)} ({database_seqs[0]}-{database_seqs[-1]})")
    logger.info(f"Query序列: {len(query_seqs)} ({query_seqs[0]}-{query_seqs[-1]})")
    logger.info(f"正样本阈值: {pos_threshold}m")
    logger.info(f"负样本阈值: {neg_threshold}m")
    logger.info("=" * 80)

    # 加载数据
    logger.info("\n[步骤1] 加载Database数据...")
    db_features, db_poses, db_seq_labels = load_bev_features_and_poses(root, database_seqs)
    logger.info(f"Database: {len(db_features)} 个特征")

    logger.info("\n[步骤2] 加载Query数据...")
    q_features, q_poses, q_seq_labels = load_bev_features_and_poses(root, query_seqs)
    logger.info(f"Query: {len(q_features)} 个特征")

    # 评估不同池化方法
    pooling_methods = ['avg', 'max']
    all_reports = {}

    for pooling in pooling_methods:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"评估池化方法: {pooling.upper()}")
        logger.info(f"{'=' * 80}")

        report = {}

        # 指标1
        metric1 = metric1_intra_inter_distance(
            db_features, db_poses, q_features, q_poses,
            pos_threshold, neg_threshold, pooling
        )
        report['metric1'] = metric1

        # 指标2
        metric2 = metric2_nn_accuracy(
            db_features, db_poses, q_features, q_poses,
            k_list=[1, 5, 10, 25], pos_threshold=pos_threshold, pooling=pooling
        )
        report['metric2'] = metric2

        # 指标3
        metric3 = metric3_clustering_quality(
            db_features, q_features, num_clusters=50, pooling=pooling
        )
        report['metric3'] = metric3

        # 指标4
        overlap = metric4_distance_distribution(
            metric1['intra_distances'], metric1['inter_distances'], pooling
        )
        report['overlap'] = overlap

        # 指标5
        metric5 = metric5_linear_separability(
            db_features, db_poses, q_features, q_poses,
            pos_threshold, neg_threshold, pooling
        )
        report['metric5'] = metric5

        all_reports[pooling] = report

    # 对比结果
    logger.info("\n\n" + "=" * 80)
    logger.info("不同池化方法对比 (使用Z补偿数据)")
    logger.info("=" * 80)

    print(f"\n{'指标':<30} {'AVG池化':<20} {'MAX池化':<20}")
    print("-" * 70)

    for metric_name, key1, key2 in [
        ('区分度 (类间/类内)', 'metric1', 'ratio'),
        ('最近邻准确率@1', 'metric2', 1),
        ('Silhouette Score', 'metric3', None),
        ('线性可分性', 'metric5', 'test_accuracy'),
        ('正负样本重叠度', 'overlap', None)
    ]:
        print(f"{metric_name:<30} ", end="")

        for pooling in pooling_methods:
            report = all_reports[pooling]

            if key2 is None:
                value = report[key1]
            else:
                value = report[key1][key2]

            if 'accuracy' in metric_name.lower() or 'overlap' in metric_name.lower():
                print(f"{value * 100:<20.2f} ", end="")
            else:
                print(f"{value:<20.4f} ", end="")

        print()

    print("=" * 80)

    # 综合评估
    logger.info("\n\n" + "=" * 80)
    logger.info("综合评估（基于AVG池化，Z补偿数据）")
    logger.info("=" * 80)

    report = all_reports['avg']
    ratio = report['metric1']['ratio']
    nn_acc = report['metric2'][1]
    silhouette = report['metric3']
    linear_acc = report['metric5']['test_accuracy']
    overlap = report['overlap']

    logger.info(f"\n关键指标:")
    logger.info(f"  特征区分度: {ratio:.4f}")
    logger.info(f"  最近邻准确率@1: {nn_acc * 100:.2f}%")
    logger.info(f"  聚类质量: {silhouette:.4f}")
    logger.info(f"  线性可分性: {linear_acc * 100:.2f}%")
    logger.info(f"  正负样本重叠度: {overlap * 100:.2f}%")

    # 评分
    score = 0
    if ratio > 1.8:
        score += 2
    elif ratio > 1.5:
        score += 1

    if nn_acc > 0.65:
        score += 2
    elif nn_acc > 0.55:
        score += 1

    if silhouette > 0.35:
        score += 2
    elif silhouette > 0.25:
        score += 1

    if linear_acc > 0.8:
        score += 2
    elif linear_acc > 0.7:
        score += 1

    logger.info(f"\n质量得分: {score}/8")

    if score >= 7:
        logger.info("✅ Stage1特征质量优秀")
        logger.info("   建议: 可以考虑优化Stage2或其他方向")
    elif score >= 5:
        logger.info("⚠️  Stage1特征质量良好但有提升空间")
        logger.info("   建议: 微调Backbone、优化损失函数")
    elif score >= 3:
        logger.info("⚠️  Stage1特征质量一般")
        logger.info("   建议: 重点改进Backbone架构和训练策略")
    else:
        logger.info("❌ Stage1特征质量较差")
        logger.info("   建议: 彻底重构Backbone或更换方法")

    logger.info("\n" + "=" * 80)
    logger.info("评估完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    # 配置路径
    config_path = '/home/wzj/pan1/BEVNet-CFPR/config/config.yml'

    # 加载配置
    config = yaml.safe_load(open(config_path))

    # 设置日志
    logger.add("/home/wzj/pan1/BEVNet-CFPR/outputs/feature_quality_cross_seq_z0.log",
               format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
               encoding='utf-8')

    # 运行评估
    evaluate_cross_sequence(config)