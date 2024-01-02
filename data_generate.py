import os
import random
from shutil import copyfile

def split_dataset(cover_path, stego_path, train_ratio=0.6, val_ratio=0.3, test_ratio=0.1):
    # 获取文件列表
    cover_files = os.listdir(cover_path)
    stego_files = os.listdir(stego_path)
    # 确保两个文件夹中文件数量相同
    assert len(cover_files) == len(stego_files), "Number of files in cover and stego folders must be the same."
    total_files = len(cover_files)
    indices = list(range(total_files))
    random.shuffle(indices)
    # 计算划分的数量
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size

    # 划分训练集
    train_indices = indices[:train_size]
    count = 1
    for i in train_indices:
        # 统一修改编码
        print(f"处理训练集 {i}.pgm:",os.path.join(cover_path, cover_files[i]), f"train/cover/{count}.pgm")
        copyfile(os.path.join(cover_path, cover_files[i]), f"train/cover/{count}.pgm")
        copyfile(os.path.join(stego_path, stego_files[i]), f"train/stego/{count}.pgm")
        count+=1

    # 划分验证集
    val_indices = indices[train_size:train_size + val_size]
    count = 1
    for i in val_indices:
        print(f"处理验证集 {i}.pgm: ",os.path.join(cover_path, cover_files[i]), f"val/cover/{count}.pgm")
        copyfile(os.path.join(cover_path, cover_files[i]), f"val/cover/{count}.pgm")
        copyfile(os.path.join(stego_path, stego_files[i]), f"val/stego/{count}.pgm")
        count +=1

    # 划分测试集
    test_indices = indices[train_size + val_size:]
    count = 1
    for i in test_indices:
        print(f"处理测试集 {i}.pgm：",os.path.join(cover_path, cover_files[i]), f"test/cover/{count}.pgm")
        copyfile(os.path.join(cover_path, cover_files[i]), f"test/cover/{count}.pgm")
        copyfile(os.path.join(stego_path, stego_files[i]), f"test/stego/{count}.pgm")
        count += 1

if __name__ == "__main__":
    # 设置路径
    cover_path = os.path.join(os.path.dirname(__file__), "images","cover")
    stego_path = os.path.join(os.path.dirname(__file__), "images","stego")

    # 创建划分后的文件夹
    os.makedirs("train/cover", exist_ok=True)
    os.makedirs("train/stego", exist_ok=True)
    os.makedirs("val/cover", exist_ok=True)
    os.makedirs("val/stego", exist_ok=True)
    os.makedirs("test/cover", exist_ok=True)
    os.makedirs("test/stego", exist_ok=True)

    # 划分数据集
    split_dataset(cover_path, stego_path)
