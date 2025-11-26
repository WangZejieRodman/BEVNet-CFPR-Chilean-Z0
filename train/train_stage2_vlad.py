"""
Stage2: 训练AttnVLADHead
使用CFPR风格的triplet loss训练全局描述符生成器
前置条件：Stage1已完成，BEV特征已提取
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

from tools.database import KittiDataset
from modules.loss import triplet_loss
from modules.net import Backbone, AttnVLADHead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
randg = np.random.RandomState()


def extract_features_if_needed(config):
    """如果BEV特征不存在，使用Stage1训练的backbone提取特征"""
    root = config["data_root"]["data_root_folder"]
    stage2_config = config["stage2_training_config"]
    training_seqs = stage2_config["training_seqs"]
    backbone_model = stage2_config["pretrained_backbone_model"]

    # 检查是否需要提取特征
    need_extract = False
    for seq in training_seqs:
        fea_folder = os.path.join(root, seq, "BEV_FEA")
        if not os.path.exists(fea_folder) or len(os.listdir(fea_folder)) == 0:
            need_extract = True
            break

    if not need_extract:
        logger.info("BEV features already exist, skipping extraction")
        return

    logger.info("BEV features not found, extracting using Stage1 backbone...")

    # 加载backbone
    from tools.utils.gen_bev_features import extract_kitti
    backbone = Backbone(32).to(device)
    checkpoint = torch.load(backbone_model)
    backbone.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(f"Backbone loaded from {backbone_model}")

    # 提取特征
    for seq in training_seqs:
        logger.info(f"Extracting BEV features for sequence {seq}")
        extract_kitti(backbone, seq, batch_num=16, root=root)

    logger.info("Feature extraction completed!")


def train_stage2(config):
    """Stage2训练主函数"""
    # 解析配置
    root = config["data_root"]["data_root_folder"]
    stage2_config = config["stage2_training_config"]
    out_folder = stage2_config["out_folder"]
    training_seqs = stage2_config["training_seqs"]
    model_name = stage2_config["model_name"]
    pretrained_vlad_model = stage2_config["pretrained_vlad_model"]
    pos_threshold = stage2_config["pos_threshold"]
    neg_threshold = stage2_config["neg_threshold"]
    batch_size = stage2_config["batch_size"]
    epochs = stage2_config["epoch"]
    lr = stage2_config["learning_rate"]
    weight_decay = stage2_config["weight_decay"]
    num_workers = stage2_config["num_workers"]

    # 创建输出目录
    os.makedirs(out_folder, exist_ok=True)

    # 设置日志
    logfile = os.path.join(out_folder, 'train_stage2.log')
    logger.add(logfile, format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}', encoding='utf-8')
    logger.info("=" * 60)
    logger.info("Stage2 Training: AttnVLADHead")
    logger.info("=" * 60)
    logger.info(f"Config: {stage2_config}")

    # 检查并提取特征（如果需要）
    extract_features_if_needed(config)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(out_folder, 'tensorboard'))

    # 创建数据集
    logger.info("Creating dataset...")
    train_dataset = KittiDataset(
        root=root,
        seqs=training_seqs,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    logger.info(f"Dataset created with {len(train_dataset)} samples")

    # 创建模型
    logger.info("Creating model...")
    model_module = __import__("modules.net", fromlist=["something"])
    vlad = getattr(model_module, model_name)().to(device=device)
    logger.info(f"Model created with {sum(p.numel() for p in vlad.parameters())} parameters")

    # 加载预训练模型（如果有）
    resume_epoch = 0
    if pretrained_vlad_model and os.path.exists(pretrained_vlad_model):
        checkpoint = torch.load(pretrained_vlad_model)
        vlad.load_state_dict(checkpoint['state_dict'], strict=False)
        try:
            resume_epoch = checkpoint.get('epoch', 0)
        except:
            resume_epoch = 0
        logger.info(f"Loaded pretrained model from {pretrained_vlad_model}, resume from epoch {resume_epoch}")

    # 优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vlad.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 损失函数
    loss_function = triplet_loss

    # 训练循环
    logger.info("Starting training...")
    step = 0

    for epoch in range(resume_epoch + 1, epochs + 1):
        vlad.train()
        epoch_loss = 0

        for i_batch, sample_batch in tqdm(enumerate(train_loader),
                                          total=len(train_loader),
                                          desc=f'Epoch {epoch}/{epochs}',
                                          leave=False):
            optimizer.zero_grad()

            # 准备输入：将query、positive、negative拼接
            input_data = torch.cat([
                sample_batch['query_desc'].flatten(0, 1),
                sample_batch['pos_desc'].flatten(0, 1),
                sample_batch['neg_desc'].flatten(0, 1),
            ], dim=0).to(device)

            # 前向传播
            out = vlad(input_data)

            # 检查输出维度
            expected_size = batch_size * 13  # 1 query + 2 pos + 10 neg
            if out.shape[0] != expected_size:
                logger.warning(f"Batch size mismatch: expected {expected_size}, got {out.shape[0]}, skipping")
                continue

            # 分离query、positive、negative特征
            query_fea, pos_fea, neg_fea = torch.split(
                out,
                [batch_size, batch_size * 2, batch_size * 10],
                dim=0
            )

            query_fea = query_fea.unsqueeze(1)
            pos_fea = pos_fea.reshape(batch_size, 2, -1)
            neg_fea = neg_fea.reshape(batch_size, 10, -1)

            # 更新数据集中的latent vectors（用于hard mining）
            train_dataset.update_latent_vectors(query_fea.squeeze(1), sample_batch['id'])

            # 计算triplet loss
            loss = loss_function(query_fea, pos_fea, neg_fea, margin=0.3)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录
            epoch_loss += loss.item()
            with torch.no_grad():
                writer.add_scalar('Loss/triplet', loss.item(), global_step=step)
                writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=step)
                step += 1

                # 定期打印
                if i_batch % 100 == 0:
                    logger.info(f"Epoch {epoch}, Batch {i_batch}/{len(train_loader)}: Loss={loss.item():.4f}")

        # Epoch结束
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} finished: Avg Loss={avg_loss:.4f}")
        writer.add_scalar('Loss/epoch_avg', avg_loss, global_step=epoch)

        # 保存模型
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(out_folder, f"epoch_{epoch}.ckpt")
            torch.save({
                'epoch': epoch,
                'state_dict': vlad.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            scheduler.step()

    # 训练完成，保存最终模型
    logger.info("Training completed!")
    final_model_path = os.path.join(out_folder, "attnvlad_final.ckpt")
    torch.save({
        'epoch': epochs,
        'state_dict': vlad.state_dict(),
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    writer.close()
    logger.info("=" * 60)
    logger.info("Stage2 Training Finished!")
    logger.info("=" * 60)


if __name__ == '__main__':
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'config/config.yml')
    config = yaml.safe_load(open(config_path))

    # 开始训练
    train_stage2(config)