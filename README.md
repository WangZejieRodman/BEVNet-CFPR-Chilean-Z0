# BEVNet-CFPR Chilean数据集训练与评估

## 项目概述

BEVNet-CFPR用于点云场景识别（Place Recognition），采用两阶段训练架构：
- **Stage1**: 训练Backbone提取BEV特征
- **Stage2**: 训练AttnVLAD聚合全局描述符

本项目使用Chilean地下矿井数据集，采用Z轴补偿方案（z0）处理地形起伏问题。

## 数据准备

### 生成Z补偿位姿文件
```bash
python tools/utils/generate_z0_csv.py
```
生成 `pointcloud_pos_ori_20m_10overlap_z0.csv`，将所有z坐标置零，后续在点云加载时实时补偿。

## 训练流程

### 1. Stage1训练（Backbone + OverlapHead）
```bash
python train/train_stage1_chilean.py
```
- 训练序列：100-159
- 输出：`outputs/stage1_chilean/backbone_final.ckpt`
- 监督信号：点对匹配、重叠度预测、高度估计

### 2. 提取BEV特征
```bash
python tools/utils/gen_bev_features_chilean.py
```
- 使用训练好的Backbone提取所有序列的BEV特征
- 输出：每个序列的 `BEV_FEA/` 文件夹

### 3. Stage2训练（AttnVLAD）
```bash
python train/train_stage2_chilean.py
```
- 训练序列：100-159
- 输出：`outputs/stage2_chilean/attnvlad_final.ckpt`
- 损失函数：Triplet Loss with Hard Mining

## 评估方法

### 整体性能评估
```bash
# 使用全局池化（跳过Stage2）
python evaluate/evaluate_chilean.py

# 使用AttnVLAD（完整Pipeline）
python evaluate/evaluate_chilean.py --use_stage2
```
- Database：160-189序列
- Query：190-209序列
- 指标：Recall@1/5/10/25

### Stage1/Stage2贡献分离

#### 1. 池化方法对比
```bash
python evaluate/evaluate_chilean_pooling.py
```
对比多种特征聚合方法：
- Global Average Pooling
- Max Pooling
- GeM Pooling
- Mixed Pooling
- AttnVLAD (Stage2)

**目的**：量化Stage2相对于简单池化的性能增益

#### 2. BEV特征质量评估
```bash
python evaluate/evaluate_feature_quality_cross_seq.py
```
从多个维度评估Stage1提取的BEV特征质量：
- 类内/类间距离比
- 最近邻准确率
- 聚类质量（Silhouette Score）
- 线性可分性
- 正负样本分布重叠度

**目的**：验证Stage1特征能否有效区分相似/不同场景

#### 3. 失败案例分析
```bash
python evaluate/analyze_failure_cases_分段.py
```
按几何距离分段（0-1m, 1-2m, ..., 9-10m），找出每段内特征距离最大的正样本对：
- 每段90个case
- 限制同一上级系列对（如20X-12X）最多出现3次
- 输出BEV 32层分层图像
- 标注Recall@K失败情况

**目的**：人工观察Stage1在困难场景下的失败模式

## 配置文件

`config/config.yml` 包含所有训练和评估参数：
- 数据路径和序列划分
- 体素化参数（坐标范围、分辨率）
- 正负样本阈值
- 训练超参数（学习率、batch size、迭代次数）
- 评估指标设置

## 关键技术点

1. **Z轴补偿**：使用z0 CSV + 实时补偿解决地形起伏
2. **跨序列评估**：Database和Query来自不同时段，避免时序偏差
3. **多级评估**：召回率 → 特征质量 → 失败案例可视化
4. **上级系列多样性**：failure case分析时保证序列对的多样性

## 输出目录结构

```
outputs/
├── stage1_chilean/
│   ├── backbone_final.ckpt
│   ├── overlap_final.ckpt
│   └── tensorboard/
├── stage2_chilean/
│   ├── attnvlad_final.ckpt
│   └── tensorboard/
└── failure_cases_segmented_diverse_z0/
    ├── 0-1m/
    ├── 1-2m/
    └── ...
```
