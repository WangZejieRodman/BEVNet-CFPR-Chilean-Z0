# tools/utils/utils.py
import open3d as o3d
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_open3d_point_cloud(xyz, color=None):
    """从坐标和颜色构建Open3D点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def load_npy_file(filename):
    """加载单个npy文件"""
    return np.load(filename)


def load_npy_files(files):
    """批量加载npy文件"""
    out = []
    for file in files:
        out.append(load_npy_file(file))
    return np.array(out)


def load_pcd(filename):
    """加载pcd格式点云"""
    pc = o3d.io.read_point_cloud(filename)
    return np.asarray(pc.points)


def load_pc_file(filename,
                 coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                 div_n=[256, 256, 32],
                 is_pcd=False,
                 sensor_z=None):
    """
    加载点云文件并体素化
    Args:
        filename: 点云文件路径
        coords_range_xyz: 体素化的xyz范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        div_n: 体素网格分辨率 [nx, ny, nz]
        is_pcd: 是否为pcd格式（否则为bin格式）
        sensor_z: 传感器Z高度，如果提供则进行Z轴补偿 (z_new = z_raw + sensor_z)
    Returns:
        voxel_out: 体素化后的occupancy grid [nx, ny, nz]
    """
    if is_pcd:
        pc = load_pcd(filename)
    else:
        # 兼容KITTI (float32, 4维) 和 Chilean (float64, 3维) 两种格式
        raw_data = np.fromfile(filename, dtype=np.float64)

        # 判断数据维度
        if raw_data.shape[0] % 3 == 0:
            # Chilean格式: float64, 3维 [x, y, z]
            pc = raw_data.reshape(-1, 3).astype(np.float32)
        elif raw_data.shape[0] % 4 == 0:
            # 可能是KITTI格式但误识别为float64，尝试float32
            raw_data_f32 = np.fromfile(filename, dtype=np.float32)
            if raw_data_f32.shape[0] % 4 == 0:
                # KITTI格式: float32, 4维 [x, y, z, intensity]
                pc = raw_data_f32.reshape(-1, 4)[:, :3]
            else:
                # 确实是float64但无法整除4，按3维处理
                pc = raw_data[:raw_data.shape[0] // 3 * 3].reshape(-1, 3).astype(np.float32)
        else:
            raise ValueError(
                f"Cannot parse point cloud file {filename}: data size {raw_data.shape[0]} cannot be divided by 3 or 4")

    # Z轴补偿（如果提供了sensor_z）
    if sensor_z is not None:
        pc[:, 2] = pc[:, 2] + sensor_z

    pc = torch.from_numpy(pc).to(device)
    ids = load_voxel(pc,
                     coords_range_xyz=coords_range_xyz,
                     div_n=div_n)
    voxel_out = torch.zeros(div_n)
    voxel_out[ids[:, 0], ids[:, 1], ids[:, 2]] = 1
    return voxel_out


def load_pc_files(files,
                  coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                  div_n=[256, 256, 32],
                  is_pcd=False,
                  sensor_z_list=None):
    """
    批量加载和体素化点云
    Args:
        files: 点云文件路径列表
        coords_range_xyz: 体素化范围
        div_n: 体素分辨率
        is_pcd: 是否为pcd格式
        sensor_z_list: 传感器Z高度列表，如果提供则进行Z轴补偿
    """
    out = []
    for i, file in enumerate(files):
        sensor_z = sensor_z_list[i] if sensor_z_list is not None else None
        out.append(load_pc_file(file, coords_range_xyz, div_n, is_pcd=is_pcd, sensor_z=sensor_z))
    return torch.stack(out)


def load_voxel(data,
               coords_range_xyz=[-10., -10, -4, 10, 10, 8],
               div_n=[256, 256, 32]):
    """
    将点云转换为体素索引
    Args:
        data: 点云坐标 [N, 3]
        coords_range_xyz: xyz范围
        div_n: 体素分辨率
    Returns:
        ids: 唯一的体素索引 [M, 3]
    """
    # 计算每个体素的尺寸
    div = [(coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0],
           (coords_range_xyz[4] - coords_range_xyz[1]) / div_n[1],
           (coords_range_xyz[5] - coords_range_xyz[2]) / div_n[2]]

    # 计算每个点所属的体素索引
    id_x = (data[:, 0] - coords_range_xyz[0]) / div[0]
    id_y = (data[:, 1] - coords_range_xyz[1]) / div[1]
    id_z = (data[:, 2] - coords_range_xyz[2]) / div[2]
    all_id = torch.cat(
        [id_x.reshape(-1, 1), id_y.reshape(-1, 1), id_z.reshape(-1, 1)], axis=1).long()

    # 过滤超出范围的点
    mask = (all_id[:, 0] >= 0) & (all_id[:, 1] >= 0) & (all_id[:, 2] >= 0) & (
            all_id[:, 0] < div_n[0]) & (all_id[:, 1] < div_n[1]) & (all_id[:, 2] < div_n[2])
    all_id = all_id[mask]
    data = data[mask]

    # 去重，返回唯一的体素索引
    ids, _, _ = torch.unique(
        all_id, return_inverse=True, return_counts=True, dim=0)

    return ids


def read_poses(file, dx=2, st=100):
    """
    读取位姿文件（用于非KITTI格式数据）
    Args:
        file: 位姿文件路径
        dx: 采样间隔（米）
        st: 跳过前st行
    Returns:
        stamp: 时间戳列表
        pose: 位姿列表 [N, 7] (x, y, z, qx, qy, qz, qw)
    """
    stamp = None
    pose = None
    delta_d = dx ** 2
    with open(file) as f:
        lines = f.readlines()[st:]
    for line in lines:
        line = line.strip().split()
        stampi = line[0]
        posei = [float(line[i]) for i in range(1, len(line))]
        if pose is None:
            pose = [posei]
            stamp = [stampi]
        else:
            # 只保留距离超过dx的帧
            diffx = posei[0] - pose[-1][0]
            diffy = posei[1] - pose[-1][1]
            if diffx ** 2 + diffy ** 2 > delta_d:
                pose.append(posei)
                stamp.append(stampi)
    pose = np.array(pose, dtype='float32')
    return stamp, pose


def se3_to_SE3(se3_pose):
    """
    将se(3)位姿转换为SE(3)变换矩阵
    Args:
        se3_pose: [x, y, z, qx, qy, qz, qw] 位姿（四元数）
    Returns:
        T: 4x4变换矩阵
    """
    translate = np.array(se3_pose[0:3], dtype='float32')
    q = se3_pose[3:]
    rot = np.array(R.from_quat(q).as_matrix(), dtype='float32')
    T = np.identity(4, dtype='float32')
    T[:3, :3] = rot
    T[:3, 3] = translate.T
    return T


def rot3d(axis, angle):
    """
    生成绕指定轴旋转的4x4变换矩阵
    Args:
        axis: 旋转轴 (0=x, 1=y, 2=z)
        angle: 旋转角度（弧度）
    Returns:
        m: 4x4变换矩阵
    """
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


def apply_transform(pts, trans):
    """
    对点云应用刚体变换
    Args:
        pts: 点云 [N, 3]
        trans: 4x4变换矩阵
    Returns:
        transformed_pts: 变换后的点云 [N, 3]
    """
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts


def occ_pcd(points, state_st=6, max_range=np.pi):
    """
    随机遮挡点云（数据增强）
    随机移除一个角度范围内的点
    Args:
        points: 点云 [N, 3]
        state_st: 随机状态阈值，>9才进行遮挡
        max_range: 最大遮挡角度范围
    Returns:
        occluded_points: 遮挡后的点云
    """
    rand_state = random.randint(state_st, 10)
    if rand_state > 9:
        # 随机选择遮挡的角度范围
        rand_start = random.uniform(-np.pi, np.pi)
        rand_end = random.uniform(rand_start, min(np.pi, rand_start + max_range))
        angles = np.arctan2(points[:, 1], points[:, 0])
        return points[(angles < rand_start) | (angles > rand_end)]
    else:
        return points


def cdist(a, b, metric='euclidean'):
    """
    计算两组点之间的距离矩阵
    Args:
        a: 第一组特征 [M, D]
        b: 第二组特征 [N, D]
        metric: 距离度量方式
    Returns:
        dist: 距离矩阵 [M, N]
    """
    if metric == 'cosine':
        return torch.sqrt(2 - 2 * torch.matmul(a, b.T))
    elif metric == 'arccosine':
        return torch.acos(torch.matmul(a, b.T))
    else:
        diffs = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)
        if metric == 'sqeuclidean':
            return torch.sum(diffs ** 2, dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
        elif metric == 'cityblock':
            return torch.sum(torch.abs(diffs), dim=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))


class AverageMeter(object):
    """计算和存储平均值、当前值、方差"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2