import shutil
from pathlib import Path

# 数据集根目录
root_dir = Path("/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale")

# 遍历所有子目录
for subdir in root_dir.iterdir():
    if subdir.is_dir():
        bev_path = subdir / "BEV_FEA"

        if bev_path.exists():
            shutil.rmtree(bev_path)
            print(f"已删除: {subdir.name}/BEV_FEA")

print("删除完成！")