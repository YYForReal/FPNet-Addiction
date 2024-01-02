# FPNet: 

这个仓库是FPNet的**PyTorch**实现（非官方）。借鉴SRNet增加了数据处理流程以及训练过程。


原始仓库: https://github.com/henryccl/FPNet


## Requirements
All experiments use the PyTorch library. We conducted training and validation on a workstation with a GTX 3080, using Ubuntu 20.04 as the operating system. We recommend installing the following package versions:

-  python=3.7 (3.11也行)

-  pytorch=1.10.0

- wandb（若需可视化）
pip install wandb

- 新建一个文件夹model存储模型文件

## 数据集
数据使用BOSSBASE 1.01的1万张灰度图像，来源于http://dde.binghamton.edu/download/。

## 隐写算法
本次复现使用的隐写算法为MiPOD算法，来源于https://dde.binghamton.edu/download/stego_algorithms/。
也可以在此网站获取到WOW隐写算法等示例代码。

matlab_MiPOD文件夹中有modified的代码，可以批处理。