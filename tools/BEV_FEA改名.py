import os
from pathlib import Path

# 数据集根目录
root_dir = Path("/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale")

# 遍历所有子目录
for subdir in root_dir.iterdir():
    if subdir.is_dir():
        old_bev_path = subdir / "BEV_FEA"
        new_bev_path = subdir / "BEV_FEA_z轴未统一"

        if old_bev_path.exists():
            old_bev_path.rename(new_bev_path)
            print(f"已重命名: {subdir.name}/BEV_FEA -> BEV_FEA_z轴未统一")

print("重命名完成！")