# evaluate/analyze_failure_cases_分段_全面类型.py
"""
分段分析Stage1 BEV特征的失败case - 保证上级系列对的多样性
按几何距离分段：0-1m, 1-2m, ..., 9-10m
每个分段找出特征距离最大的前90个case
限制：同一对上级系列（如20X-12X）最多出现3次
标注该case在Recall@K中的失败情况
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
from tqdm import tqdm
from loguru import logger
import cv2
from collections import defaultdict


def get_upper_series(seq_id):
    """
    获取序列的上级系列（前两位）

    Args:
        seq_id: 序列ID，如 "204", "123"
    Returns:
        upper_series: 上级系列，如 "20", "12"
    """
    return seq_id[:2] if len(seq_id) >= 2 else seq_id


def load_bev_features_and_info(root, seqs):
    """
    加载BEV特征、几何位置、文件路径（使用z0 CSV）

    Returns:
        features: BEV特征列表 [(512, 32, 32), ...]
        poses: 几何位置列表 [(x, y), ...]
        file_paths: 点云文件路径列表
        seq_ids: 序列ID列表
        sensor_z_dict: {seq: {timestamp: sensor_z}}
    """
    features = []
    poses = []
    file_paths = []
    seq_ids = []
    sensor_z_dict = {}

    for seq in tqdm(seqs, desc="加载数据"):
        fea_folder = os.path.join(root, seq, "BEV_FEA")
        pc_folder = os.path.join(root, seq, "pointcloud_20m_10overlap")
        pose_file = os.path.join(root, seq, "pointcloud_pos_ori_20m_10overlap_z0.csv")
        pose_file_original = os.path.join(root, seq, "pointcloud_pos_ori_20m_10overlap.csv")

        if not os.path.exists(fea_folder) or not os.path.exists(pose_file):
            continue

        # 读取z0位置数据
        df = pd.read_csv(pose_file)
        pose_dict = {}
        for _, row in df.iterrows():
            timestamp = str(int(row['timestamp']))
            x, y = row['x'], row['y']
            pose_dict[timestamp] = (x, y)

        # 读取原始sensor_z
        df_original = pd.read_csv(pose_file_original)
        sensor_z_dict_seq = {}
        for _, row in df_original.iterrows():
            timestamp = str(int(row['timestamp']))
            sensor_z = row['z']
            sensor_z_dict_seq[timestamp] = sensor_z
        sensor_z_dict[seq] = sensor_z_dict_seq

        # 读取特征
        fea_files = sorted([f for f in os.listdir(fea_folder) if f.endswith('.npy')])

        for fea_file in fea_files:
            timestamp = fea_file.replace('.npy', '')

            if timestamp not in pose_dict:
                continue

            # 特征路径
            fea_path = os.path.join(fea_folder, fea_file)
            fea = np.load(fea_path)  # (512, 32, 32)

            # 点云路径
            pc_path = os.path.join(pc_folder, f"{timestamp}.bin")

            features.append(fea)
            poses.append(pose_dict[timestamp])
            file_paths.append(pc_path)
            seq_ids.append(seq)

    return features, poses, file_paths, seq_ids, sensor_z_dict


def pooling_features(features, method='avg'):
    """对BEV特征进行池化"""
    descriptors = []

    for fea in features:
        if method == 'avg':
            desc = fea.mean(axis=(1, 2))  # (512,)
        elif method == 'max':
            desc = fea.max(axis=(1, 2))
        else:
            raise ValueError(f"Unknown pooling: {method}")

        # L2归一化
        desc = desc / (np.linalg.norm(desc) + 1e-8)
        descriptors.append(desc)

    return np.array(descriptors)


def compute_feature_distance(desc1, desc2):
    """计算两个描述符之间的欧氏距离"""
    return np.linalg.norm(desc1 - desc2)


def compute_recall_status(query_descriptors, database_descriptors, ground_truth, top_k_list=[1, 5, 10, 25]):
    """
    计算每个query在各个Recall@K的状态

    Returns:
        recall_status: dict {query_idx: {k: True/False}}
    """
    logger.info("计算Recall状态...")

    # 计算相似度矩阵
    similarity = query_descriptors @ database_descriptors.T  # (M, N)

    # 对每个query，找到最相似的top_k个database
    max_k = max(top_k_list)
    top_indices = np.argsort(-similarity, axis=1)[:, :max_k]  # (M, max_k)

    # 计算每个query在各个K值的成功/失败状态
    recall_status = {}

    for query_idx, positives in ground_truth.items():
        if len(positives) == 0:
            # 没有正样本，跳过
            continue

        status = {}
        for k in top_k_list:
            # 检查top-k中是否有正样本
            top_k_preds = top_indices[query_idx, :k]
            status[k] = any(pred in positives for pred in top_k_preds)

        recall_status[query_idx] = status

    return recall_status


def find_failure_cases_segmented_diverse(db_features, db_poses, db_paths, db_seq_ids,
                                         q_features, q_poses, q_paths, q_seq_ids,
                                         recall_status,
                                         segments=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                                                   (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)],
                                         target_cases_per_segment=90,
                                         max_per_upper_series_pair=3,
                                         pooling='avg'):
    """
    分段查找失败case - 保证上级系列对的多样性

    Args:
        segments: 几何距离分段列表，如 [(0, 1), (1, 2), ...]
        target_cases_per_segment: 每个分段目标case数量
        max_per_upper_series_pair: 同一对上级系列最多出现次数
        recall_status: 每个query的Recall@K状态

    Returns:
        segmented_cases: dict {(min_dist, max_dist): [cases]}
    """
    logger.info(f"\n分段查找失败case (保证上级系列对多样性)...")
    logger.info(f"分段: {segments}")
    logger.info(f"每段目标case数: {target_cases_per_segment}")
    logger.info(f"同一上级系列对最多出现: {max_per_upper_series_pair}次")

    # 池化
    db_desc = pooling_features(db_features, pooling)
    q_desc = pooling_features(q_features, pooling)

    # 为每个分段收集正样本对
    segment_pairs = {seg: [] for seg in segments}

    for i in tqdm(range(len(q_poses)), desc="收集正样本对"):
        q_pos = np.array(q_poses[i])

        # 获取该query的Recall状态（如果没有则跳过）
        if i not in recall_status:
            continue

        for j in range(len(db_poses)):
            db_pos = np.array(db_poses[j])

            # 计算几何距离
            geo_dist = np.linalg.norm(q_pos - db_pos)

            # 找到对应的分段
            for seg_min, seg_max in segments:
                if seg_min <= geo_dist < seg_max:
                    # 计算特征距离
                    fea_dist = compute_feature_distance(q_desc[i], db_desc[j])

                    # 确定在哪些Recall@K中失败
                    failed_at = []
                    for k in [1, 5, 10, 25]:
                        if not recall_status[i][k]:
                            failed_at.append(k)

                    # 获取上级系列
                    q_upper = get_upper_series(q_seq_ids[i])
                    db_upper = get_upper_series(db_seq_ids[j])
                    upper_series_pair = (q_upper, db_upper)

                    segment_pairs[(seg_min, seg_max)].append({
                        'query_idx': i,
                        'db_idx': j,
                        'query_path': q_paths[i],
                        'db_path': db_paths[j],
                        'query_seq': q_seq_ids[i],
                        'db_seq': db_seq_ids[j],
                        'query_upper_series': q_upper,
                        'db_upper_series': db_upper,
                        'upper_series_pair': upper_series_pair,
                        'geo_distance': geo_dist,
                        'feature_distance': fea_dist,
                        'query_feature': q_features[i],  # (512, 32, 32)
                        'db_feature': db_features[j],
                        'failed_at_k': failed_at  # 在哪些Recall@K中失败
                    })
                    break

    # 对每个分段按特征距离排序，并应用上级系列对多样性约束
    segmented_cases = {}

    for seg, pairs in segment_pairs.items():
        logger.info(f"\n分段 [{seg[0]:.0f}m, {seg[1]:.0f}m): 共 {len(pairs)} 个正样本对")

        if len(pairs) == 0:
            segmented_cases[seg] = []
            continue

        # 按特征距离降序排序
        pairs.sort(key=lambda x: x['feature_distance'], reverse=True)

        # 应用上级系列对多样性约束
        selected_cases = []
        upper_series_pair_count = defaultdict(int)  # 记录每对上级系列已选数量

        for case in pairs:
            upper_pair = case['upper_series_pair']

            # 检查该上级系列对是否已达到上限
            if upper_series_pair_count[upper_pair] < max_per_upper_series_pair:
                selected_cases.append(case)
                upper_series_pair_count[upper_pair] += 1

                # 达到目标数量则停止
                if len(selected_cases) >= target_cases_per_segment:
                    break

        logger.info(f"  从{len(pairs)}个候选中选出{len(selected_cases)}个case")
        logger.info(f"  涉及{len(upper_series_pair_count)}对不同的上级系列")

        # 统计上级系列对分布
        if len(upper_series_pair_count) > 0:
            logger.info(f"  上级系列对分布示例:")
            sorted_pairs = sorted(upper_series_pair_count.items(),
                                  key=lambda x: x[1], reverse=True)[:90]
            for upper_pair, count in sorted_pairs:
                logger.info(f"    {upper_pair[0]}X - {upper_pair[1]}X: {count}次")

        if len(selected_cases) > 0:
            logger.info(f"  特征距离范围: "
                        f"{selected_cases[-1]['feature_distance']:.4f} ~ "
                        f"{selected_cases[0]['feature_distance']:.4f}")

        segmented_cases[seg] = selected_cases

    return segmented_cases


def load_and_voxelize_pointcloud(pc_path, sensor_z, coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                                 div_n=[256, 256, 32]):
    """加载点云并体素化为occupancy grid（应用Z补偿）"""
    # 读取点云
    raw_data = np.fromfile(pc_path, dtype=np.float64)

    if raw_data.shape[0] % 3 != 0:
        raise ValueError(f"Invalid point cloud file {pc_path}")

    points = raw_data.reshape(-1, 3).astype(np.float32)

    # Z轴补偿
    points[:, 2] = points[:, 2] + sensor_z

    # 计算体素索引
    div = [(coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0],
           (coords_range_xyz[4] - coords_range_xyz[1]) / div_n[1],
           (coords_range_xyz[5] - coords_range_xyz[2]) / div_n[2]]

    id_x = (points[:, 0] - coords_range_xyz[0]) / div[0]
    id_y = (points[:, 1] - coords_range_xyz[1]) / div[1]
    id_z = (points[:, 2] - coords_range_xyz[2]) / div[2]

    all_id = np.stack([id_x, id_y, id_z], axis=1).astype(np.int32)

    # 过滤超出范围的点
    mask = (all_id[:, 0] >= 0) & (all_id[:, 1] >= 0) & (all_id[:, 2] >= 0) & \
           (all_id[:, 0] < div_n[0]) & (all_id[:, 1] < div_n[1]) & (all_id[:, 2] < div_n[2])

    all_id = all_id[mask]

    # 构建occupancy grid
    voxel_grid = np.zeros(div_n, dtype=np.uint8)
    voxel_grid[all_id[:, 0], all_id[:, 1], all_id[:, 2]] = 1

    return voxel_grid


def save_bev_layers_as_images(voxel_grid, output_folder):
    """保存occupancy grid的32层为png图像"""
    os.makedirs(output_folder, exist_ok=True)

    # 遍历32层
    for z in range(voxel_grid.shape[2]):
        # 提取第z层 (256, 256)
        layer = voxel_grid[:, :, z]

        # 转换为0-255图像（0=黑色，255=白色）
        img = (layer * 255).astype(np.uint8)

        # 保存（z从下到上，层0是最底层-4m附近，层31是最顶层8m附近）
        img_path = os.path.join(output_folder, f"layer_{z:02d}.png")
        cv2.imwrite(img_path, img)


def build_ground_truth(query_poses, database_poses, pos_threshold):
    """构建Ground Truth（用于计算Recall状态）"""
    ground_truth = {}

    for i, q_pos in enumerate(query_poses):
        # 计算到所有database的距离
        distances = np.linalg.norm(database_poses - q_pos, axis=1)

        # 找到正样本
        positives = np.where(distances < pos_threshold)[0].tolist()

        ground_truth[i] = positives

    return ground_truth


def analyze_failure_cases_segmented_diverse(config):
    """主函数：分段分析失败case - 保证上级系列对多样性"""
    root = config["data_root"]["data_root_folder"]

    # Database和Query序列
    database_seqs = [str(i) for i in range(100, 160)]
    query_seqs = [str(i) for i in range(160, 210)]

    # 参数
    segments = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    target_cases_per_segment = 90  # 每段目标90个case
    max_per_upper_series_pair = 3  # 同一上级系列对最多3次
    pooling = 'avg'
    pos_threshold_for_recall = 10.0  # 用于计算Recall@K的正样本阈值

    logger.info("=" * 80)
    logger.info("Stage1 BEV特征失败case分段分析 (保证上级系列对多样性)")
    logger.info("=" * 80)
    logger.info(f"几何距离分段: {segments}")
    logger.info(f"每段目标case数: {target_cases_per_segment}")
    logger.info(f"同一上级系列对最多: {max_per_upper_series_pair}次")
    logger.info(f"池化方式: {pooling}")
    logger.info(f"Recall@K正样本阈值: {pos_threshold_for_recall}m")
    logger.info("=" * 80)

    # 加载Database数据
    logger.info("\n[步骤1] 加载Database数据...")
    db_features, db_poses, db_paths, db_seq_ids, db_sensor_z = load_bev_features_and_info(root, database_seqs)
    logger.info(f"Database: {len(db_features)} 个点云")

    # 加载Query数据
    logger.info("\n[步骤2] 加载Query数据...")
    q_features, q_poses, q_paths, q_seq_ids, q_sensor_z = load_bev_features_and_info(root, query_seqs)
    logger.info(f"Query: {len(q_features)} 个点云")

    # 计算Recall@K状态
    logger.info("\n[步骤3] 计算Recall@K状态...")
    db_poses_array = np.array(db_poses)
    q_poses_array = np.array(q_poses)
    ground_truth = build_ground_truth(q_poses_array, db_poses_array, pos_threshold_for_recall)

    db_desc = pooling_features(db_features, pooling)
    q_desc = pooling_features(q_features, pooling)

    recall_status = compute_recall_status(q_desc, db_desc, ground_truth, top_k_list=[1, 5, 10, 25])
    logger.info(f"计算了 {len(recall_status)} 个query的Recall状态")

    # 分段查找失败case - 应用上级系列对多样性约束
    logger.info("\n[步骤4] 分段查找失败case (保证上级系列对多样性)...")
    segmented_cases = find_failure_cases_segmented_diverse(
        db_features, db_poses, db_paths, db_seq_ids,
        q_features, q_poses, q_paths, q_seq_ids,
        recall_status,
        segments=segments,
        target_cases_per_segment=target_cases_per_segment,
        max_per_upper_series_pair=max_per_upper_series_pair,
        pooling=pooling
    )

    # 创建输出目录
    output_root = "/home/wzj/pan1/BEVNet-CFPR/outputs/failure_cases_segmented_diverse_z0"
    os.makedirs(output_root, exist_ok=True)

    # 生成报告并可视化
    logger.info(f"\n[步骤5] 生成报告和BEV分层可视化...")

    # 为每个分段创建报告
    all_report_data = []

    for seg, cases in segmented_cases.items():
        seg_min, seg_max = seg
        seg_name = f"{seg_min:.0f}-{seg_max:.0f}m"

        if len(cases) == 0:
            logger.info(f"\n分段 [{seg_name}]: 无正样本对，跳过")
            continue

        logger.info(f"\n分段 [{seg_name}]: 处理 {len(cases)} 个case")

        # 创建分段文件夹
        seg_folder = os.path.join(output_root, seg_name)
        os.makedirs(seg_folder, exist_ok=True)

        for rank, case in enumerate(tqdm(cases, desc=f"处理 {seg_name}", leave=False)):
            case_id = f"case_{rank:03d}"

            # 创建该case的文件夹
            case_folder = os.path.join(seg_folder, case_id)
            os.makedirs(case_folder, exist_ok=True)

            try:
                # 获取sensor_z
                q_timestamp = os.path.basename(case['query_path']).replace('.bin', '')
                db_timestamp = os.path.basename(case['db_path']).replace('.bin', '')
                q_sz = q_sensor_z[case['query_seq']][q_timestamp]
                db_sz = db_sensor_z[case['db_seq']][db_timestamp]

                # Query BEV分层可视化
                query_bev_folder = os.path.join(case_folder, "query_bev_layers")
                q_voxel = load_and_voxelize_pointcloud(case['query_path'], q_sz)
                save_bev_layers_as_images(q_voxel, query_bev_folder)

                # Database BEV分层可视化
                db_bev_folder = os.path.join(case_folder, "db_bev_layers")
                db_voxel = load_and_voxelize_pointcloud(case['db_path'], db_sz)
                save_bev_layers_as_images(db_voxel, db_bev_folder)

                # 格式化Recall@K失败信息
                if len(case['failed_at_k']) == 0:
                    failed_info = "成功"
                else:
                    failed_info = "失败于 R@" + ",".join([str(k) for k in sorted(case['failed_at_k'])])

                # 记录到报告
                all_report_data.append({
                    'segment': seg_name,
                    'rank_in_segment': rank + 1,
                    'case_id': case_id,
                    'query_seq': case['query_seq'],
                    'query_upper_series': case['query_upper_series'],
                    'db_seq': case['db_seq'],
                    'db_upper_series': case['db_upper_series'],
                    'upper_series_pair': f"{case['query_upper_series']}X-{case['db_upper_series']}X",
                    'query_path': case['query_path'],
                    'query_bev_folder': query_bev_folder,
                    'db_path': case['db_path'],
                    'db_bev_folder': db_bev_folder,
                    'geo_distance': f"{case['geo_distance']:.3f}",
                    'feature_distance': f"{case['feature_distance']:.4f}",
                    'recall_status': failed_info
                })

            except Exception as e:
                logger.error(f"  处理 {seg_name}/{case_id} 失败: {e}")
                continue

    # 保存总体CSV报告
    report_df = pd.DataFrame(all_report_data)
    report_csv = os.path.join(output_root, "failure_cases_segmented_diverse_report.csv")
    report_df.to_csv(report_csv, index=False)

    logger.info(f"\n报告已保存: {report_csv}")
    logger.info(f"BEV分层图已保存在: {output_root}/")

    # 打印每个分段的前3个case
    logger.info("\n" + "=" * 80)
    logger.info("每个分段的前3个最差case:")
    logger.info("=" * 80)

    for seg in segments:
        seg_name = f"{seg[0]:.0f}-{seg[1]:.0f}m"
        seg_cases = [c for c in all_report_data if c['segment'] == seg_name]

        if len(seg_cases) == 0:
            continue

        logger.info(f"\n【分段 {seg_name}】")
        for i in range(min(3, len(seg_cases))):
            case = seg_cases[i]
            logger.info(f"  Case {i + 1}:")
            logger.info(f"    上级系列对: {case['upper_series_pair']}")
            logger.info(f"    Query:  {case['query_seq']} ({case['query_path']})")
            logger.info(f"    DB:     {case['db_seq']} ({case['db_path']})")
            logger.info(f"    几何距离: {case['geo_distance']}m")
            logger.info(f"    特征距离: {case['feature_distance']}")
            logger.info(f"    Recall状态: {case['recall_status']}")

    # 统计信息
    logger.info("\n" + "=" * 80)
    logger.info("统计信息:")
    logger.info("=" * 80)

    for seg in segments:
        seg_name = f"{seg[0]:.0f}-{seg[1]:.0f}m"
        seg_cases = [c for c in all_report_data if c['segment'] == seg_name]

        if len(seg_cases) == 0:
            logger.info(f"  {seg_name}: 无正样本对")
            continue

        # 统计上级系列对分布
        upper_series_pair_count = defaultdict(int)
        for case in seg_cases:
            upper_series_pair_count[case['upper_series_pair']] += 1

        # 统计Recall@K失败情况
        failed_counts = {1: 0, 5: 0, 10: 0, 25: 0}
        for case in seg_cases:
            if case['recall_status'] != "成功":
                # 解析失败的K值
                failed_str = case['recall_status'].replace("失败于 R@", "")
                for k_str in failed_str.split(","):
                    k = int(k_str)
                    failed_counts[k] += 1

        logger.info(f"\n  {seg_name}: {len(seg_cases)} 个case")
        logger.info(f"    涉及 {len(upper_series_pair_count)} 对不同上级系列")
        logger.info(f"    上级系列对分布 (前30):")
        sorted_pairs = sorted(upper_series_pair_count.items(),
                              key=lambda x: x[1], reverse=True)[:30]
        for upper_pair, count in sorted_pairs:
            logger.info(f"      {upper_pair}: {count}次")
        logger.info(f"    失败于 R@1:  {failed_counts[1]}/{len(seg_cases)}")
        logger.info(f"    失败于 R@5:  {failed_counts[5]}/{len(seg_cases)}")
        logger.info(f"    失败于 R@10: {failed_counts[10]}/{len(seg_cases)}")
        logger.info(f"    失败于 R@25: {failed_counts[25]}/{len(seg_cases)}")

    logger.info("\n" + "=" * 80)
    logger.info("分段分析完成 (已保证上级系列对多样性)！")
    logger.info("=" * 80)


if __name__ == '__main__':
    # 配置路径
    config_path = '/home/wzj/pan1/BEVNet-CFPR/config/config.yml'

    # 加载配置
    config = yaml.safe_load(open(config_path))

    # 设置日志
    logger.add("/home/wzj/pan1/BEVNet-CFPR/outputs/failure_cases_segmented_diverse_analysis_z0.log",
               format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
               encoding='utf-8')

    # 运行分段分析
    analyze_failure_cases_segmented_diverse(config)