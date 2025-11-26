from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import open3d as o3d
import pathlib


class KittiDataset(Dataset):
    """
    CFPR的Stage2数据集：用于训练AttnVLAD
    加载query-positive-negative triplets用于triplet loss训练
    """

    def __init__(self, root, seqs, pos_threshold, neg_threshold) -> None:
        super().__init__()
        self.root = root
        self.seqs = seqs
        self.poses = []
        self.fea_cache = {}

        # 加载所有序列的位姿
        for seq in seqs:
            pose = np.genfromtxt(os.path.join(root, seq, 'poses.txt'))[:, [3, 11]]
            self.poses.append(pose)

        self.pairs = {}
        self.randg = np.random.RandomState()

        # 构建pairs字典：为每一帧找到正样本和负样本候选
        key = 0
        acc_num = 0
        for i in range(len(self.poses)):
            pose = self.poses[i]
            # 计算帧间距离矩阵
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose ** 2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))
            # 正样本：距离<pos_threshold且不是自己
            id_pos = np.argwhere((dis < pos_threshold) & (dis > 0))
            # 负样本候选：距离<neg_threshold（用于构建all_ids集合）
            id_neg = np.argwhere(dis < neg_threshold)
            for j in range(len(pose)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": set(negatives.tolist())}
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
            id = str(query["query_id"]).zfill(6)

            fea_file = os.path.join(self.root, seq, "BEV_FEA", id + '.npy')
            self.fea_cache[idx] = torch.from_numpy(np.load(fea_file))

    def get_random_positive(self, idx, num):
        """随机采样num个正样本"""
        positives = self.pairs[idx]["positives"]
        randid = np.random.randint(0, len(positives), num).tolist()
        return [positives[i] for i in randid]

    def get_random_negative(self, idx, num):
        """随机采样num个负样本（不在negatives集合中的帧）"""
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        randid = np.random.randint(0, len(negatives), num).tolist()
        return [negatives[i] for i in randid]

    def get_random_hard_positive(self, idx, num):
        """
        采样hard positive：在正样本中选择描述符距离最远的
        如果还没有描述符，退化为随机采样
        """
        query_vec = self.traing_latent_vectors[idx]
        if query_vec.sum() == 0:
            return self.get_random_positive(idx, num)

        random_pos = self.pairs[idx]["positives"]
        random_pos = torch.Tensor(random_pos).long()
        latent_vecs = self.traing_latent_vectors[random_pos]
        mask = latent_vecs.sum(dim=1) != 0
        latent_vecs = latent_vecs[mask]
        query_vec = self.traing_latent_vectors[idx].unsqueeze(0)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)
        maxid = torch.argsort(diff)[-num:]
        return random_pos[maxid].tolist()

    def get_random_hard_negative(self, idx, num):
        """
        采样hard negative：在负样本中选择描述符距离最近的
        如果还没有描述符，退化为随机采样
        """
        query_vec = self.traing_latent_vectors[idx]
        if query_vec.sum() == 0:
            return self.get_random_negative(idx, num)

        random_neg = list(self.all_ids - self.pairs[idx]["negatives"])
        random_neg = torch.Tensor(random_neg).long()
        latent_vecs = self.traing_latent_vectors[random_neg]
        mask = latent_vecs.sum(dim=1) != 0
        latent_vecs = latent_vecs[mask]
        query_vec = self.traing_latent_vectors[idx].unsqueeze(0)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)
        minid = torch.argsort(diff)[:num]
        return random_neg[minid].tolist()

    def get_other_neg(self, id_pos, id_neg):
        """获取与正样本和负样本都不同的其他负样本（用于quadruplet loss）"""
        random_neg = list(self.all_ids - self.pairs[id_pos]["negatives"] - self.pairs[id_neg]["negatives"])
        randid = np.random.randint(0, len(random_neg) - 1)
        return random_neg[randid]

    def update_latent_vectors(self, fea, idx):
        """更新训练过程中的全局描述符缓存，用于hard mining"""
        for i in range(len(idx)):
            self.traing_latent_vectors[idx[i]] = fea[i]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pos_num = 2  # 每个query采样2个正样本
        neg_num = 10  # 每个query采样10个负样本

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


class KITTIDatasetOverlap(Dataset):
    """
    BEVNet风格的Stage1数据集：用于训练Backbone + OverlapHead
    加载query-positive点云对，计算ICP变换，用于点级监督训练
    """

    def __init__(self,
                 sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                 root="/media/l/yp2/KITTI/odometry/dataset/sequences/",
                 pos_threshold_min=10,
                 pos_threshold_max=20,
                 neg_thresgold=50,
                 coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                 div_n=[256, 256, 32],
                 random_rotation=True,
                 random_occ=False,
                 num_iter=300000) -> None:
        super().__init__()
        self.num_iter = num_iter
        # ICP结果缓存目录
        self.icp_path = root.replace("sequences", "icp")
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)
        self.div_n = div_n
        self.coords_range_xyz = coords_range_xyz
        self.kitti_icp_cache = {}  # 内存缓存ICP结果
        self.random_rotation = random_rotation
        self.random_occ = random_occ
        self.device = torch.device('cpu')
        self.randg = np.random.RandomState()
        self.root = root
        self.sequs = sequs

        # 加载所有序列的位姿
        self.poses = []
        for seq in sequs:
            pose = np.genfromtxt(os.path.join(root, seq, 'poses.txt'))
            self.poses.append(pose)

        # 构建正负样本pairs
        key = 0
        acc_num = 0
        self.pairs = {}
        for i in range(len(self.poses)):
            pose = self.poses[i][:, [3, 11]]  # 只取x, z坐标
            # 计算距离矩阵
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose ** 2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))
            # 正样本：pos_threshold_min < 距离 < pos_threshold_max
            id_pos = np.argwhere((dis < pos_threshold_max) & (dis > pos_threshold_min))
            # 负样本候选区域
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

    @property
    def velo2cam(self):
        """KITTI的雷达到相机的标定矩阵"""
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02,
                7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03,
                1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_random_positive(self, idx):
        """随机采样一个正样本"""
        positives = self.pairs[idx]["positives"]
        randid = random.randint(0, len(positives) - 1)
        return positives[randid]

    def get_random_negative(self, idx):
        """随机采样一个负样本"""
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        randid = random.randint(0, len(negatives) - 1)
        return negatives[randid]

    def load_pcd(self, idx):
        """加载指定索引的点云"""
        query = self.pairs[idx]
        seq = self.sequs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        file = os.path.join(self.root, seq, "velodyne", id + '.bin')
        return np.fromfile(file, dtype='float32').reshape(-1, 4)[:, 0:3]

    def get_icp_name(self, query_id, pos_id):
        """生成ICP结果的缓存文件名"""
        query = self.pairs[query_id]
        drive = int(self.sequs[query["query_seq"]])
        t0 = query["query_id"]
        pos = self.pairs[pos_id]
        t1 = pos["query_id"]
        key = '%d_%d_%d' % (drive, t0, t1)
        return os.path.join(self.icp_path, key + '.npy')

    def get_odometry(self, idx):
        """获取指定索引的位姿"""
        query = self.pairs[idx]
        T_w_cam0 = self.poses[query["query_seq"]][query["query_id"]].reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 采样query和positive
        queryid = idx % len(self.pairs)
        posid = self.get_random_positive(queryid)
        query_points = self.load_pcd(queryid)
        pos_points = self.load_pcd(posid)
        query_odom = self.get_odometry(queryid)
        pos_odom = self.get_odometry(posid)

        # 计算或加载ICP变换
        filename = self.get_icp_name(queryid, posid)
        if filename not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                # 使用odometry初始化，然后运行ICP精细化
                M = (self.velo2cam @ query_odom.T @ np.linalg.inv(pos_odom.T)
                     @ np.linalg.inv(self.velo2cam)).T
                query_points_t = self.apply_transform(query_points, M)
                pcd0 = self.make_open3d_point_cloud(query_points_t)
                pcd1 = self.make_open3d_point_cloud(pos_points)
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                try:
                    M2 = np.load(filename)
                except Exception as inst:
                    print(inst)
                    # 重新计算
                    M = (self.velo2cam @ query_odom.T @ np.linalg.inv(pos_odom.T)
                         @ np.linalg.inv(self.velo2cam)).T
                    query_points_t = self.apply_transform(query_points, M)
                    pcd0 = self.make_open3d_point_cloud(query_points_t)
                    pcd1 = self.make_open3d_point_cloud(pos_points)
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd0, pcd1, 0.2, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                    M2 = M @ reg.transformation
                    np.save(filename, M2)
            self.kitti_icp_cache[filename] = M2
        else:
            M2 = self.kitti_icp_cache[filename]

        # 数据增强：随机旋转
        if self.random_rotation:
            T0 = self.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            T1 = self.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            trans = T1 @ M2 @ np.linalg.inv(T0)
            query_points = self.apply_transform(query_points, T0)
            pos_points = self.apply_transform(pos_points, T1)
        else:
            trans = M2

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


if __name__ == "__main__":
    # 测试数据加载
    dataset = KITTIDatasetOverlap(
        sequs=['00'],
        root="/path/to/kitti/sequences",
        pos_threshold_min=0,
        pos_threshold_max=60,
        neg_thresgold=140,
        coords_range_xyz=[-50., -50, -4, 50, 50, 3],
        div_n=[256, 256, 32],
        random_rotation=True,
        random_occ=True,
        num_iter=100)

    for i in range(10):
        d = dataset[random.randint(0, len(dataset) - 1)]
        print(f"Sample {i}: voxel0 shape={d['voxel0'].shape}, "
              f"points0 shape={d['points0'].shape}")