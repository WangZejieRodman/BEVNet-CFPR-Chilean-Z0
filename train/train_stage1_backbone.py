"""
Stage1: 训练Backbone + OverlapHead
使用BEVNet风格的点级监督训练
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tools.database import KITTIDatasetOverlap
from modules.net import BEVNet
from modules.loss_stage1 import pair_loss, overlap_loss, dist_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_stage1(config):
    """Stage1训练主函数"""
    # 解析配置
    root = config["data_root"]["data_root_folder"]
    stage1_config = config["stage1_training_config"]
    out_folder = stage1_config["out_folder"]
    training_seqs = stage1_config["training_seqs"]
    coords_range_xyz = stage1_config["coords_range_xyz"]
    div_n = stage1_config["div_n"]
    batch_size = stage1_config["batch_size"]
    num_iter = stage1_config["num_iter"]
    lr = stage1_config["learning_rate"]
    weight_decay = stage1_config["weight_decay"]
    lr_decay_gamma = stage1_config["lr_decay_gamma"]
    lr_decay_step = stage1_config["lr_decay_step"]
    save_interval = stage1_config["save_interval"]
    num_workers = stage1_config["num_workers"]

    # 创建输出目录
    os.makedirs(out_folder, exist_ok=True)

    # 设置日志
    logfile = os.path.join(out_folder, 'train_stage1.log')
    logger.add(logfile, format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}', encoding='utf-8')
    logger.info("=" * 60)
    logger.info("Stage1 Training: Backbone + OverlapHead")
    logger.info("=" * 60)
    logger.info(f"Config: {stage1_config}")

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(out_folder, 'tensorboard'))

    # 创建数据集
    logger.info("Creating dataset...")
    train_dataset = KITTIDatasetOverlap(
        sequs=training_seqs,
        root=root,
        pos_threshold_min=stage1_config["pos_threshold_min"],
        pos_threshold_max=stage1_config["pos_threshold_max"],
        neg_thresgold=stage1_config["neg_threshold"],
        coords_range_xyz=coords_range_xyz,
        div_n=div_n,
        random_rotation=stage1_config["random_rotation"],
        random_occ=stage1_config["random_occlusion"],
        num_iter=num_iter
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    logger.info(f"Dataset created with {len(train_dataset)} iterations")

    # 创建模型
    logger.info("Creating model...")
    model = BEVNet(div_n[2]).to(device=device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)

    logger.info("Starting training...")
    model.train()

    ### 从头训练
    step = 0

    ### 断点续训
    # resume_step = 0
    # resume_ckpt = "/home/wzj/pan1/CFPR-master/outputs/stage1/checkpoint_step_1000.ckpt"  # ← 改成需要的
    #
    # if os.path.exists(resume_ckpt):
    #     logger.info(f"Resuming from {resume_ckpt}")
    #     checkpoint = torch.load(resume_ckpt)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     resume_step = checkpoint['step']
    #     logger.info(f"Resumed from step {resume_step}")
    #
    # step = resume_step

    for i_batch, sample_batch in tqdm(enumerate(train_loader),
                                      total=len(train_loader),
                                      desc='Stage1 Training'):
        try:
            optimizer.zero_grad()

            # 准备输入：将两个点云拼接成batch
            input_voxel = torch.cat([
                sample_batch['voxel0'],
                sample_batch['voxel1']
            ], dim=0).to(device)

            # 前向传播
            out, out4, x4 = model(input_voxel)

            # 分离两个点云的特征
            mask1 = (out.indices[:, 0] == 0)
            mask2 = (out.indices[:, 0] == 1)

            # 计算损失
            # 1. Pair loss: 点对匹配损失
            total_loss, desc_loss, det_loss, score_loss, z_loss, z_loss0, z_loss1, correct_ratio = pair_loss(
                out.features[mask1, :],
                out.features[mask2, :],
                sample_batch['trans0'][0],
                sample_batch['trans1'][0],
                sample_batch['points0'][0],
                sample_batch['points1'][0],
                sample_batch['points_xy0'][0],
                sample_batch['points_xy1'][0],
                num_height=div_n[2],
                min_z=coords_range_xyz[2],
                height=coords_range_xyz[5] - coords_range_xyz[2],
                search_radiu=max((coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0], 0.3)
            )

            # 2. Overlap loss: 重叠区域预测损失
            loss4, precision4, recall4 = overlap_loss(
                out4,
                sample_batch['trans0'][0],
                sample_batch['trans1'][0],
                coords_range_xyz[0],
                coords_range_xyz[3],
                show=False
            )

            # 3. Dist loss: 特征距离损失（在x4层级）
            desc_loss4, acc4 = dist_loss(
                x4,
                sample_batch['trans0'][0],
                sample_batch['trans1'][0],
                coords_range_xyz[0],
                coords_range_xyz[3]
            )

            # 组合损失
            if desc_loss4 is not None:
                loss_all = desc_loss4 + loss4
            else:
                loss_all = loss4

            if total_loss is not None:
                loss_all = loss_all + total_loss

            # 反向传播
            loss_all.backward()
            optimizer.step()

            # 记录日志
            with torch.no_grad():
                writer.add_scalar('Loss/total', loss_all.item(), global_step=step)
                writer.add_scalar('Loss/overlap', loss4.item(), global_step=step)
                writer.add_scalar('Metrics/overlap_precision', precision4, global_step=step)
                writer.add_scalar('Metrics/overlap_recall', recall4, global_step=step)
                writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=step)

                if total_loss is not None:
                    writer.add_scalar('Loss/desc', desc_loss.item(), global_step=step)
                    writer.add_scalar('Loss/det', det_loss.item(), global_step=step)
                    writer.add_scalar('Loss/score', score_loss.item(), global_step=step)
                    writer.add_scalar('Loss/z', z_loss.item(), global_step=step)
                    writer.add_scalar('Loss/z0', z_loss0.item(), global_step=step)
                    writer.add_scalar('Loss/z1', z_loss1.item(), global_step=step)
                    writer.add_scalar('Metrics/match_accuracy', correct_ratio, global_step=step)

                if desc_loss4 is not None:
                    writer.add_scalar('Loss/desc4', desc_loss4.item(), global_step=step)
                    writer.add_scalar('Metrics/acc4', acc4, global_step=step)

                # 定期打印
                if step % 100 == 0:
                    logger.info(f"Step {step}/{num_iter}: Loss={loss_all.item():.4f}, "
                                f"Overlap Loss={loss4.item():.4f}, "
                                f"Precision={precision4:.4f}, Recall={recall4:.4f}")

                step += 1

            # 保存模型
            if step % save_interval == 0:
                checkpoint_path = os.path.join(out_folder, f"checkpoint_step_{step}.ckpt")
                torch.save({
                    'step': step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

            # 学习率衰减
            if step % lr_decay_step == 0 and step > 0:
                scheduler.step()
                logger.info(f"Learning rate decayed to {optimizer.state_dict()['param_groups'][0]['lr']}")

        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 训练完成，保存最终模型
    logger.info("Training completed!")

    # 保存完整BEVNet模型
    final_model_path = os.path.join(out_folder, "bevnet_final.ckpt")
    torch.save({
        'step': step,
        'state_dict': model.state_dict(),
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # 分离保存Backbone和OverlapHead
    logger.info("Extracting Backbone and OverlapHead...")

    # Backbone权重（包含所有下采样层）
    backbone_state_dict = {}
    for name, param in model.state_dict().items():
        if name.startswith('dconv_down') or name.startswith('maxpool'):
            backbone_state_dict[name] = param

    backbone_path = os.path.join(out_folder, "backbone_final.ckpt")
    torch.save({'state_dict': backbone_state_dict}, backbone_path)
    logger.info(f"Backbone saved to {backbone_path}")

    # OverlapHead权重
    overlap_state_dict = {}
    for name, param in model.state_dict().items():
        if name.startswith('fusenet16') or name.startswith('last_conv16'):
            overlap_state_dict[name] = param

    overlap_path = os.path.join(out_folder, "overlap_final.ckpt")
    torch.save({'state_dict': overlap_state_dict}, overlap_path)
    logger.info(f"OverlapHead saved to {overlap_path}")

    writer.close()
    logger.info("=" * 60)
    logger.info("Stage1 Training Finished!")
    logger.info("=" * 60)


if __name__ == '__main__':
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'config/config.yml')
    config = yaml.safe_load(open(config_path))

    # 开始训练
    train_stage1(config)