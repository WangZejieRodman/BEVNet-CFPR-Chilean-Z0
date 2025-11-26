# tools/utils/generate_z0_csv.py
"""
生成Z轴补偿后的位姿CSV文件
将所有序列的 pointcloud_pos_ori_20m_10overlap.csv 中的 z 列置为 0
生成 pointcloud_pos_ori_20m_10overlap_z0.csv
"""
import os
import pandas as pd
from tqdm import tqdm
from loguru import logger

# 配置
data_root = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale"

# 所有序列 (100-209)
all_seqs = [str(i) for i in range(100, 210)]

# 统计信息
stats = {
    'total': 0,
    'success': 0,
    'skipped': 0,
    'failed': 0
}

logger.add("generate_z0_csv.log",
           format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
           encoding='utf-8')

logger.info("=" * 80)
logger.info("生成Z轴补偿后的位姿CSV文件")
logger.info("=" * 80)
logger.info(f"数据根目录: {data_root}")
logger.info(f"待处理序列: {len(all_seqs)} (100-209)")
logger.info("=" * 80)

for seq in tqdm(all_seqs, desc="处理序列"):
    stats['total'] += 1

    # 原始CSV路径
    csv_path = os.path.join(data_root, seq, "pointcloud_pos_ori_20m_10overlap.csv")

    # 输出CSV路径
    csv_z0_path = os.path.join(data_root, seq, "pointcloud_pos_ori_20m_10overlap_z0.csv")

    # 检查原始CSV是否存在
    if not os.path.exists(csv_path):
        logger.warning(f"序列 {seq}: CSV不存在，跳过")
        stats['skipped'] += 1
        continue

    # 检查是否已存在z0 CSV
    if os.path.exists(csv_z0_path):
        logger.info(f"序列 {seq}: z0 CSV已存在，跳过")
        stats['skipped'] += 1
        continue

    try:
        # 读取原始CSV
        df = pd.read_csv(csv_path)

        # 检查是否有z列
        if 'z' not in df.columns:
            logger.error(f"序列 {seq}: CSV中没有z列")
            stats['failed'] += 1
            continue

        # 记录原始z的范围（用于验证）
        z_min = df['z'].min()
        z_max = df['z'].max()
        z_mean = df['z'].mean()

        # 创建副本并将z列置为0
        df_z0 = df.copy()
        df_z0['z'] = 0.0

        # 保存新CSV
        df_z0.to_csv(csv_z0_path, index=False)

        logger.info(f"序列 {seq}: 成功生成 (原始z范围: [{z_min:.3f}, {z_max:.3f}], 均值: {z_mean:.3f})")
        stats['success'] += 1

    except Exception as e:
        logger.error(f"序列 {seq}: 处理失败 - {e}")
        stats['failed'] += 1

# 输出统计信息
logger.info("\n" + "=" * 80)
logger.info("处理完成！统计信息:")
logger.info("=" * 80)
logger.info(f"总序列数:   {stats['total']}")
logger.info(f"成功生成:   {stats['success']}")
logger.info(f"跳过:       {stats['skipped']}")
logger.info(f"失败:       {stats['failed']}")
logger.info("=" * 80)

if stats['success'] > 0:
    logger.info(f"\n✓ 成功生成 {stats['success']} 个序列的 z0 CSV文件")
    logger.info(f"文件命名: pointcloud_pos_ori_20m_10overlap_z0.csv")
    logger.info(f"\n下一步: 修改数据加载代码，使用 z0 CSV 并实时补偿点云")
else:
    logger.warning(f"\n⚠ 没有成功生成任何 z0 CSV 文件，请检查数据路径")