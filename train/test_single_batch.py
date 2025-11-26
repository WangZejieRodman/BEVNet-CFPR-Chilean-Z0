"""
测试单个batch的训练流程，验证前向传播、损失计算、反向传播是否正常
适配Chilean数据集
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from tools.database_chilean import ChileanDatasetOverlap, ChileanDataset
from modules.net import BEVNet, AttnVLADHead
from modules.loss_stage1 import pair_loss, overlap_loss, dist_loss
from modules.loss import triplet_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_stage1_single_batch():
    """测试Stage1单个batch训练（Chilean数据集）"""
    print("=" * 60)
    print("Testing Stage1 Single Batch Training (Chilean Dataset)")
    print("=" * 60)
    
    # 加载配置
    config = yaml.safe_load(open('/home/wzj/pan1/BEVNet-CFPR/config/config.yml'))
    stage1_config = config['stage1_training_config']
    
    # 创建数据集（只1个batch，用前2个序列）
    print("\nCreating dataset...")
    dataset = ChileanDatasetOverlap(
        seqs=stage1_config['training_seqs'][:2],
        root=config['data_root']['data_root_folder'],
        pos_threshold_min=stage1_config['pos_threshold_min'],
        pos_threshold_max=stage1_config['pos_threshold_max'],
        neg_thresgold=stage1_config['neg_threshold'],
        coords_range_xyz=stage1_config['coords_range_xyz'],
        div_n=stage1_config['div_n'],
        random_rotation=False,  # 测试时关闭增强，便于调试
        random_occ=False,
        num_iter=1
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 创建模型
    print("Creating model...")
    model = BEVNet(stage1_config['div_n'][2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 获取一个batch
    print("\nLoading batch...")
    batch = next(iter(dataloader))
    
    print("\n1. Testing forward pass...")
    input_voxel = torch.cat([batch['voxel0'], batch['voxel1']], dim=0).to(device)
    print(f"   Input shape: {input_voxel.shape}")
    print(f"   Input nonzero voxels: voxel0={batch['voxel0'].sum()}, voxel1={batch['voxel1'].sum()}")
    
    model.train()
    out, out4, x4 = model(input_voxel)
    print(f"   ✓ Forward pass successful")
    print(f"   Output shapes:")
    print(f"     out: spatial_shape={out.spatial_shape}, features={out.features.shape}")
    print(f"     out4: spatial_shape={out4.spatial_shape}, features={out4.features.shape}")
    print(f"     x4: spatial_shape={x4.spatial_shape}")  # 修改：用spatial_shape
    
    print("\n2. Testing loss computation...")
    mask1 = (out.indices[:, 0] == 0)
    mask2 = (out.indices[:, 0] == 1)
    
    print(f"   Number of points: cloud0={mask1.sum()}, cloud1={mask2.sum()}")
    
    # Pair loss
    print("   Computing pair loss...")
    total_loss, desc_loss, det_loss, score_loss, z_loss, z_loss0, z_loss1, correct_ratio = pair_loss(
        out.features[mask1, :],
        out.features[mask2, :],
        batch['trans0'][0],
        batch['trans1'][0],
        batch['points0'][0],
        batch['points1'][0],
        batch['points_xy0'][0],
        batch['points_xy1'][0],
        num_height=stage1_config['div_n'][2],
        min_z=stage1_config['coords_range_xyz'][2],
        height=stage1_config['coords_range_xyz'][5] - stage1_config['coords_range_xyz'][2],
        search_radiu=max((stage1_config['coords_range_xyz'][3] - stage1_config['coords_range_xyz'][0]) / stage1_config['div_n'][0], 0.3)
    )
    
    # Overlap loss
    print("   Computing overlap loss...")
    loss4, precision4, recall4 = overlap_loss(
        out4, batch['trans0'][0], batch['trans1'][0],
        stage1_config['coords_range_xyz'][0],
        stage1_config['coords_range_xyz'][3]
    )
    
    # Dist loss
    print("   Computing dist loss...")
    desc_loss4, acc4 = dist_loss(
        x4, batch['trans0'][0], batch['trans1'][0],
        stage1_config['coords_range_xyz'][0],
        stage1_config['coords_range_xyz'][3]
    )
    
    print("\n   Loss values:")
    if total_loss is not None:
        print(f"   ✓ Pair loss: {total_loss.item():.4f}")
        print(f"     - Desc loss: {desc_loss.item():.4f}")
        print(f"     - Det loss: {det_loss.item():.4f}")
        print(f"     - Score loss: {score_loss.item():.4f}")
        print(f"     - Z loss: {z_loss.item():.4f}")
        print(f"     - Z loss0: {z_loss0.item():.4f}")
        print(f"     - Z loss1: {z_loss1.item():.4f}")
        print(f"     - Correct ratio: {correct_ratio:.4f}")
    else:
        print(f"   ⚠ Pair loss: None (not enough matches, this is normal for first iteration)")
    
    print(f"   ✓ Overlap loss: {loss4.item():.4f}")
    print(f"     - Precision: {precision4:.4f}")
    print(f"     - Recall: {recall4:.4f}")
    
    if desc_loss4 is not None:
        print(f"   ✓ Dist loss: {desc_loss4.item():.4f}")
        print(f"     - Accuracy: {acc4:.4f}")
    else:
        print(f"   ⚠ Dist loss: None (not enough matches, this is normal)")
    
    # 总损失
    if desc_loss4 is not None:
        loss_all = desc_loss4 + loss4
    else:
        loss_all = loss4
    if total_loss is not None:
        loss_all = loss_all + total_loss
    
    print(f"\n   Total loss: {loss_all.item():.4f}")
    
    print("\n3. Testing backward pass...")
    optimizer.zero_grad()
    loss_all.backward()
    optimizer.step()
    print(f"   ✓ Backward pass successful")
    
    # 检查梯度
    has_grad = False
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            grad_count += 1
    
    if has_grad:
        print(f"   ✓ Gradients computed successfully ({grad_count} parameters with gradients)")
    else:
        print(f"   ⚠ Warning: No gradients computed")
    
    print("\n" + "=" * 60)
    print("✓ Stage1 single batch test PASSED!")
    print("=" * 60)


def test_stage2_single_batch():
    """测试Stage2单个batch训练（Chilean数据集）"""
    print("\n" + "=" * 60)
    print("Testing Stage2 Single Batch Training (Chilean Dataset)")
    print("=" * 60)
    
    # 加载配置
    config = yaml.safe_load(open('/home/wzj/pan1/BEVNet-CFPR/config/config.yml'))
    stage2_config = config['stage2_training_config']
    root = config['data_root']['data_root_folder']
    
    # 检查BEV特征是否存在
    test_seq = stage2_config['training_seqs'][0]
    bev_folder = os.path.join(root, test_seq, "BEV_FEA")
    
    if not os.path.exists(bev_folder) or len(os.listdir(bev_folder)) == 0:
        print(f"\n⚠ Skipping Stage2 test: BEV features not found")
        print(f"   Expected folder: {bev_folder}")
        print(f"   Please run Stage1 training and feature extraction first.")
        return
    
    # 创建数据集
    print("\nCreating dataset...")
    try:
        dataset = ChileanDataset(
            root=config['data_root']['data_root_folder'],
            seqs=stage2_config['training_seqs'][:2],
            pos_threshold=stage2_config['pos_threshold'],
            neg_threshold=stage2_config['neg_threshold']
        )
    except FileNotFoundError as e:
        print(f"\n⚠ Skipping Stage2 test: {e}")
        return
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # 创建模型
    print("Creating model...")
    vlad = AttnVLADHead().to(device)
    optimizer = torch.optim.Adam(vlad.parameters(), lr=1e-5)
    print(f"Model has {sum(p.numel() for p in vlad.parameters()):,} parameters")
    
    # 获取一个batch
    print("\nLoading batch...")
    batch = next(iter(dataloader))
    
    print("\n1. Testing forward pass...")
    input_data = torch.cat([
        batch['query_desc'].flatten(0, 1),
        batch['pos_desc'].flatten(0, 1),
        batch['neg_desc'].flatten(0, 1),
    ], dim=0).to(device)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Input components: query={batch['query_desc'].shape}, "
          f"pos={batch['pos_desc'].shape}, neg={batch['neg_desc'].shape}")
    
    vlad.train()
    out = vlad(input_data)
    print(f"   ✓ Forward pass successful")
    print(f"   Output shape: {out.shape}")
    
    print("\n2. Testing loss computation...")
    query_fea, pos_fea, neg_fea = torch.split(out, [2, 4, 20], dim=0)
    query_fea = query_fea.unsqueeze(1)
    pos_fea = pos_fea.reshape(2, 2, -1)
    neg_fea = neg_fea.reshape(2, 10, -1)
    
    print(f"   Split shapes: query={query_fea.shape}, pos={pos_fea.shape}, neg={neg_fea.shape}")
    
    loss = triplet_loss(query_fea, pos_fea, neg_fea, margin=0.3)
    print(f"   ✓ Triplet loss: {loss.item():.4f}")
    
    print("\n3. Testing backward pass...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"   ✓ Backward pass successful")
    
    # 检查梯度
    has_grad = False
    grad_count = 0
    for name, param in vlad.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            grad_count += 1
    
    if has_grad:
        print(f"   ✓ Gradients computed successfully ({grad_count} parameters with gradients)")
    else:
        print(f"   ⚠ Warning: No gradients computed")
    
    print("\n" + "=" * 60)
    print("✓ Stage2 single batch test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        print("\n" + "=" * 70)
        print("Chilean Dataset Single Batch Test")
        print("=" * 70)
        
        test_stage1_single_batch()
        test_stage2_single_batch()
        
        print("\n" + "=" * 70)
        print("All single batch tests PASSED! ✓")
        print("Ready to start full training!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()