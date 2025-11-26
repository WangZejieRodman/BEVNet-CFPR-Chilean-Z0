# tools/database_chilean.py
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import open3d as o3d
import pathlib
import pandas as pd


class ChileanDatasetOverlap(Dataset):
    """
    Chilean数据集的Stage1数据集：用于训练Backbone + OverlapHead
    加载query-positive点云对，使用真值位姿计算变换（无需ICP）
    """

    def __init__(self,
                 seqs=['100', '101', '102'],
                 root="/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale/",
                 pos_threshold_min=0,
                 pos_threshold_max=7,
                 neg_thresgold=30,
                 coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                 div_n=[256, 256, 32],
                 random_rotation=True,
                 random_occ=False,
                 num_iter=300000) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.div_n = div_n
        self.coords_range_xyz = coords_range_xyz
        self.random_rotation = random_rotation
        self.random_occ = random_occ
        self.device = torch.device('cpu')
        self.randg = np.random.RandomState()
        self.root = root
        self.seqs = seqs

        # 加载所有序列的位姿（从z0 CSV文件读取）
        self.poses = []
        self.timestamps = []  # 存储每个序列的timestamp列表
        self.sensor_z_dict = {}  # 存储每个序列每个timestamp的原始sensor_z

        for seq in seqs:
            # 使用z0 CSV文件路径
            pose_file = os.path.join(root, seq, 'pointcloud_pos_ori_20m_10overlap_z0.csv')

            # 同时读取原始CSV以获取sensor_z
            pose_file_original = os.path.join(root, seq, 'pointcloud_pos_ori_20m_10overlap.csv')

            # 读取z0 CSV：用于计算距离
            df = pd.read_csv(pose_file, sep=',')
            # 提取xy坐标用于计算距离
            pose_xy = df[['x', 'y']].values.astype(np.float32)
            self.poses.append(pose_xy)
            self.timestamps.append(df['timestamp'].values)

            # 读取原始CSV：获取sensor_z用于点云补偿
            df_original = pd.read_csv(pose_file_original, sep=',')
            sensor_z_dict_seq = {}
            for _, row in df_original.iterrows():
                timestamp = str(int(row['timestamp']))
                sensor_z = row['z']
                sensor_z_dict_seq[timestamp] = sensor_z
            self.sensor_z_dict[seq] = sensor_z_dict_seq

        # 构建正负样本pairs
        key = 0
        acc_num = 0
        self.pairs = {}
        for i in range(len(self.poses)):
            pose = self.poses[i]
            # 计算距离矩阵（只用xy平面距离）
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose ** 2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))

            # 正样本：pos_threshold_min < 距离 < pos_threshold_max
            id_pos = np.argwhere((dis < pos_threshold_max) & (dis > pos_threshold_min))
            # 负样本候选区域：距离 < neg_threshold
            id_neg = np.argwhere(dis < neg_thresgold)

            for j in range(len(pose)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": set(negatives.tolist())
                }
                key += 1
            acc_num += len(pose)
        self.all_ids = set(range(len(self.pairs)))

        # 预加载所有位姿的SE3矩阵（使用z0位姿）
        self.poses_SE3 = []
        for i, seq in enumerate(seqs):
            pose_file = os.path.join(root, seq, 'pointcloud_pos_ori_20m_10overlap_z0.csv')
            df = pd.read_csv(pose_file, sep=',')

            # 转换为SE3矩阵
            seq_poses = []
            for _, row in df.iterrows():
                se3 = [row['x'], row['y'], row['z'],  # z已经是0了
                       row['qx'], row['qy'], row['qz'], row['qw']]
                T = self.se3_to_SE3(se3)
                seq_poses.append(T)
            self.poses_SE3.append(seq_poses)

    def se3_to_SE3(self, se3_pose):
        """将se(3)位姿转换为SE(3)变换矩阵"""
        from scipy.spatial.transform import Rotation as R
        translate = np.array(se3_pose[0:3], dtype='float32')
        q = se3_pose[3:]
        rot = np.array(R.from_quat(q).as_matrix(), dtype='float32')
        T = np.identity(4, dtype='float32')
        T[:3, :3] = rot
        T[:3, 3] = translate.T
        return T

    def get_random_positive(self, idx):
        """随机采样一个正样本"""
        positives = self.pairs[idx]["positives"]
        if len(positives) == 0:
            return None
        randid = random.randint(0, len(positives) - 1)
        return positives[randid]

    def get_random_negative(self, idx):
        """随机采样一个负样本"""
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        if len(negatives) == 0:
            return None
        randid = random.randint(0, len(negatives) - 1)
        return negatives[randid]

    def load_pcd(self, idx):
        """
        加载指定索引的点云，并进行Z轴补偿
        """
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        timestamp = self.timestamps[query["query_seq"]][query["query_id"]]

        # Chilean格式：pointcloud_20m_10overlap/timestamp.bin
        file = os.path.join(self.root, seq, "pointcloud_20m_10overlap", f"{timestamp}.bin")

        # 读取float64格式，转为float32
        raw_data = np.fromfile(file, dtype=np.float64)
        if raw_data.shape[0] % 3 != 0:
            raise ValueError(f"Invalid point cloud file {file}: size {raw_data.shape[0]} not divisible by 3")
        pc = raw_data.reshape(-1, 3).astype(np.float32)

        # Z轴补偿
        sensor_z = self.sensor_z_dict[seq][str(timestamp)]
        pc[:, 2] = pc[:, 2] + sensor_z

        return pc

    def get_odometry(self, idx):
        """获取指定索引的位姿（SE3矩阵，已经是z0版本）"""
        query = self.pairs[idx]
        T = self.poses_SE3[query["query_seq"]][query["query_id"]]
        return T

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 采样query和positive
        queryid = idx % len(self.pairs)
        posid = self.get_random_positive(queryid)

        # 如果没有正样本，重新采样
        if posid is None:
            queryid = random.randint(0, len(self.pairs) - 1)
            posid = self.get_random_positive(queryid)
            if posid is None:
                # 实在找不到，用自己作为正样本
                posid = queryid

        query_points = self.load_pcd(queryid)
        pos_points = self.load_pcd(posid)
        query_odom = self.get_odometry(queryid)
        pos_odom = self.get_odometry(posid)

        # 直接使用真值位姿计算变换（无需ICP）
        # chilean_NoRot_NoScale数据集只用平移，不用旋转（点云已对齐）
        query_translation = query_odom[:3, 3]
        pos_translation = pos_odom[:3, 3]
        trans = np.identity(4, dtype='float32')
        trans[:3, 3] = query_translation - pos_translation

        # 数据增强：随机旋转
        if self.random_rotation:
            T0 = self.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            T1 = self.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            trans = T1 @ trans @ np.linalg.inv(T0)
            query_points = self.apply_transform(query_points, T0)
            pos_points = self.apply_transform(pos_points, T1)
        else:
            trans = trans

        # 数据增强：随机遮挡
        if self.random_occ:
            query_points = self.occ_pcd(query_points, state_st=6, max_range=np.pi)
            pos_points = self.occ_pcd(pos_points, state_st=6, max_range=np.pi)

        # 体素化
        ids0, points0, ids_xy0, points_xy0 = self.load_voxel(
            query_points, self.coords_range_xyz, self.div_n)
        ids1, points1, ids_xy1, points_xy1 = self.load_voxel(
            pos_points, self.coords_range_xyz, self.div_n)

        voxel_out0 = np.zeros(self.div_n, dtype='float32')
        voxel_out0[ids0[:, 0], ids0[:, 1], ids0[:, 2]] = 1
        voxel_out1 = np.zeros(self.div_n, dtype='float32')
        voxel_out1[ids1[:, 0], ids1[:, 1], ids1[:, 2]] = 1

        return {
            "voxel0": voxel_out0,
            "voxel1": voxel_out1,
            "trans0": trans.astype('float32'),
            "trans1": np.identity(4, dtype='float32'),
            "points0": points0,
            "points1": points1,
            "points_xy0": points_xy0,
            "points_xy1": points_xy1
        }

    def make_open3d_point_cloud(self, xyz, color=None):
        """构建Open3D点云对象"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd

    def rot3d(self, axis, angle):
        """生成3D旋转矩阵"""
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(4)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

    def apply_transform(self, pts, trans):
        """应用刚体变换"""
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def occ_pcd(self, points, state_st=6, max_range=np.pi):
        """随机遮挡点云（数据增强）"""
        rand_state = random.randint(state_st, 10)
        if rand_state > 9:
            rand_start = random.uniform(-np.pi, np.pi)
            rand_end = random.uniform(rand_start, min(np.pi, rand_start + max_range))
            angles = np.arctan2(points[:, 1], points[:, 0])
            return points[(angles < rand_start) | (angles > rand_end)]
        else:
            return points

    def load_voxel(self, data, coords_range_xyz, div_n):
        """
        将点云体素化并返回有用的信息
        Returns:
            ids: 唯一的3D体素索引 [M, 3]
            pooled_data: 每个体素的平均3D坐标 [M, 3]
            ids_xy: 唯一的2D体素索引 [N, 2]
            pooled_data_xy: 每个2D体素的平均xy坐标 [N, 2]
        """
        import torch_scatter

        div = [(coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0],
               (coords_range_xyz[4] - coords_range_xyz[1]) / div_n[1],
               (coords_range_xyz[5] - coords_range_xyz[2]) / div_n[2]]
        id_x = (data[:, 0] - coords_range_xyz[0]) / div[0]
        id_y = (data[:, 1] - coords_range_xyz[1]) / div[1]
        id_z = (data[:, 2] - coords_range_xyz[2]) / div[2]
        all_id = np.concatenate(
            [id_x.reshape(-1, 1), id_y.reshape(-1, 1), id_z.reshape(-1, 1)],
            axis=1).astype('int32')

        # 过滤超出范围的点
        mask = (all_id[:, 0] >= 0) & (all_id[:, 1] >= 0) & (all_id[:, 2] >= 0) & (
                all_id[:, 0] < div_n[0]) & (all_id[:, 1] < div_n[1]) & (all_id[:, 2] < div_n[2])
        all_id = all_id[mask]
        data = data[mask]

        # 3D体素化
        all_id_torch = torch.from_numpy(all_id).long().to(self.device)
        ids, unq_inv, _ = torch.unique(all_id_torch, return_inverse=True,
                                       return_counts=True, dim=0)
        ids = ids.detach().cpu().numpy().astype('int32')
        pooled_data = torch_scatter.scatter_mean(
            torch.from_numpy(data).to(self.device), unq_inv, dim=0)

        # 2D体素化（BEV）
        ids_xy, unq_inv_xy, _ = torch.unique(all_id_torch[:, :2],
                                             return_inverse=True,
                                             return_counts=True, dim=0)
        ids_xy = ids_xy.detach().cpu().numpy().astype('int32')
        pooled_data_xy = torch_scatter.scatter_mean(
            torch.from_numpy(data[:, :2]).to(self.device), unq_inv_xy, dim=0)

        return ids, pooled_data.detach().cpu().numpy(), ids_xy, pooled_data_xy.detach().cpu().numpy()


class ChileanDataset(Dataset):
    """
    Chilean数据集的Stage2数据集：用于训练AttnVLAD
    加载query-positive-negative triplets用于triplet loss训练
    """

    def __init__(self, root, seqs, pos_threshold, neg_threshold) -> None:
        super().__init__()
        self.root = root
        self.seqs = seqs
        self.poses = []
        self.fea_cache = {}
        self.timestamps = []

        # 加载所有序列的位姿（从z0 CSV）
        for seq in seqs:
            pose_file = os.path.join(root, seq, 'pointcloud_pos_ori_20m_10overlap_z0.csv')
            df = pd.read_csv(pose_file, sep=',')

            # 只用xy坐标计算距离
            pose_xy = df[['x', 'y']].values.astype(np.float32)
            self.poses.append(pose_xy)
            self.timestamps.append(df['timestamp'].values)

        self.pairs = {}
        self.randg = np.random.RandomState()

        # 构建pairs字典：为每一帧找到正样本和负样本候选
        key = 0
        acc_num = 0
        for i in range(len(self.poses)):
            pose = self.poses[i]
            # 计算帧间距离矩阵（xy平面）
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose ** 2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))

            # 正样本：距离<pos_threshold且不是自己
            id_pos = np.argwhere((dis < pos_threshold) & (dis > 0))
            # 负样本候选：距离>neg_threshold
            id_neg = np.argwhere(dis > neg_threshold)

            for j in range(len(pose)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": negatives.tolist()
                }
                key += 1
            acc_num += len(pose)

        self.all_ids = set(range(len(self.pairs)))
        # 用于存储训练过程中的全局描述符，用于hard mining
        self.traing_latent_vectors = torch.zeros((len(self.pairs), 1024))

        # 预加载所有BEV特征到内存
        self.load_all_features()

    def load_all_features(self):
        """预加载所有BEV特征文件到内存"""
        for idx in tqdm(range(len(self.pairs)), desc="Loading features"):
            query = self.pairs[idx]
            seq = self.seqs[query["query_seq"]]
            timestamp = self.timestamps[query["query_seq"]][query["query_id"]]

            # BEV特征保存在序列目录下的BEV_FEA文件夹
            fea_file = os.path.join(self.root, seq, "BEV_FEA", f"{timestamp}.npy")

            if not os.path.exists(fea_file):
                raise FileNotFoundError(f"BEV feature file not found: {fea_file}. "
                                        f"Please run Stage1 training and feature extraction first.")

            self.fea_cache[idx] = torch.from_numpy(np.load(fea_file))

    def get_random_positive(self, idx, num):
        """随机采样num个正样本"""
        positives = self.pairs[idx]["positives"]
        if len(positives) == 0:
            return [idx] * num
        if len(positives) < num:
            randid = np.random.choice(len(positives), num, replace=True).tolist()
        else:
            randid = np.random.choice(len(positives), num, replace=False).tolist()
        return [positives[i] for i in randid]

    def get_random_negative(self, idx, num):
        """随机采样num个负样本"""
        negatives = self.pairs[idx]["negatives"]
        if len(negatives) == 0:
            negatives = list(self.all_ids - {idx})
        if len(negatives) < num:
            randid = np.random.choice(len(negatives), num, replace=True).tolist()
        else:
            randid = np.random.choice(len(negatives), num, replace=False).tolist()
        return [negatives[i] for i in randid]

    def get_random_hard_positive(self, idx, num):
        """采样hard positive"""
        query_vec = self.traing_latent_vectors[idx]
        if query_vec.sum() == 0:
            return self.get_random_positive(idx, num)

        random_pos = self.pairs[idx]["positives"]
        if len(random_pos) == 0:
            return [idx] * num

        random_pos = torch.Tensor(random_pos).long()
        latent_vecs = self.traing_latent_vectors[random_pos]
        mask = latent_vecs.sum(dim=1) != 0

        if mask.sum() == 0:
            return self.get_random_positive(idx, num)

        latent_vecs = latent_vecs[mask]
        random_pos = random_pos[mask]

        query_vec = self.traing_latent_vectors[idx].unsqueeze(0)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)

        if len(diff) < num:
            result = random_pos.tolist()
            result = (result * ((num // len(result)) + 1))[:num]
            return result

        maxid = torch.argsort(diff)[-num:]
        return random_pos[maxid].tolist()

    def get_random_hard_negative(self, idx, num):
        """采样hard negative"""
        query_vec = self.traing_latent_vectors[idx]
        if query_vec.sum() == 0:
            return self.get_random_negative(idx, num)

        random_neg = self.pairs[idx]["negatives"]
        if len(random_neg) == 0:
            random_neg = list(self.all_ids - {idx})

        random_neg = torch.Tensor(random_neg).long()
        latent_vecs = self.traing_latent_vectors[random_neg]
        mask = latent_vecs.sum(dim=1) != 0

        if mask.sum() == 0:
            return self.get_random_negative(idx, num)

        latent_vecs = latent_vecs[mask]
        random_neg = random_neg[mask]

        query_vec = self.traing_latent_vectors[idx].unsqueeze(0)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)

        if len(diff) < num:
            result = random_neg.tolist()
            result = (result * ((num // len(result)) + 1))[:num]
            return result

        minid = torch.argsort(diff)[:num]
        return random_neg[minid].tolist()

    def update_latent_vectors(self, fea, idx):
        """更新训练过程中的全局描述符缓存，用于hard mining"""
        for i in range(len(idx)):
            self.traing_latent_vectors[idx[i]] = fea[i]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pos_num = 2
        neg_num = 10

        queryid = idx % len(self.pairs)
        posid = self.get_random_hard_positive(queryid, pos_num)
        negid = self.get_random_hard_negative(queryid, neg_num)

        # 获取缓存的特征
        query_fea = self.fea_cache[queryid].unsqueeze(0)

        pos_feas = torch.zeros((pos_num, 512, 32, 32))
        for i in range(pos_num):
            pos_feas[i] = self.fea_cache[posid[i]]

        neg_feas = torch.zeros((neg_num, 512, 32, 32))
        for i in range(neg_num):
            neg_feas[i] = self.fea_cache[negid[i]]

        return {
            "id": queryid,
            "query_desc": query_fea,
            "pos_desc": pos_feas,
            "neg_desc": neg_feas,
        }