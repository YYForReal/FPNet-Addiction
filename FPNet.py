import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from PIL import Image

SRM_npy = np.load(os.path.join(os.path.dirname(__file__), 'SRM_Kernels.npy'))


# 定义SRMConv2d类，继承nn.Module
class SRMConv2d(nn.Module):

    # 初始化函数，设置输入通道数、输出通道数、卷积核大小、步长、填充
    def __init__(self, stride=1, padding=0):
        super(SRMConv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        # 初始化权重和偏置
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), requires_grad=True)
        # 初始化权重和偏置
        self.reset_parameters()


    # 初始化权重和偏置
    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    # 定义前向传播函数，使用F.conv2d函数，计算卷积结果
    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ConvModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Conv_layer(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1):
        super(Conv_layer, self).__init__()
        if k == 1:
            self.conv = nn.Conv2d(in_dim, out_dim, k, stride=s)
        elif k == 3:
            self.conv = nn.Conv2d(in_dim, out_dim, k, padding=1, stride=s)

        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# 特征传递块（FPB）类
class FPB(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5):
        super(FPB, self).__init__()
        # 计算中间通道数mid_channels
        mid_channels = int(out_dim * expand_ratio)

        # 主卷积层，用于处理输入特征
        self.main_conv = Conv_layer(in_dim, mid_channels, 1)
        # 短卷积层，用于短连接
        self.short_conv = Conv_layer(in_dim, mid_channels, 1)
        # 最终卷积层，将处理后的特征拼接并输出
        self.final_conv = Conv_layer(mid_channels*2, out_dim, 1)
        # 两个额外的卷积层，用于对主卷积层的特征进行处理
        self.conv1 = Conv_layer(mid_channels, mid_channels, 1)
        self.conv2 = Conv_layer(mid_channels, mid_channels, 3)

    def forward(self, x):
        # 主卷积层的处理
        x_main = self.main_conv(x)
        # 短卷积层
        x_short = self.short_conv(x)
        # 保留主卷积层的输出，用于后续与处理后的主卷积层输出相加
        res = x_main
        # 主卷积层的额外处理
        x_main = self.conv1(x_main)
        x_main = self.conv2(x_main)
        # 将主卷积层的输出与保留的输出相加
        x_main = x_main + res
        # 将短卷积层的输出和主卷积层的输出在通道维度上拼接
        x = torch.cat([x_short, x_main], dim=1)
        # 最终卷积层的处理
        x = self.final_conv(x)
        # 返回处理后的特征
        return x


class ADM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ADM, self).__init__()
        self.conv = ConvModule(in_channels * 4, out_channels)  # 使用 ConvModule 实现卷积操作
        self.att_conv = nn.Conv2d(in_channels, in_channels, 1) # 用于学习通道注意力的卷积层
        self.act = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_att = x.mean((2, 3), keepdim=True)# 计算输入特征图 x 的通道维度的平均值，得到注意力权重
        x_att = self.att_conv(x_att)
        x_att = self.act(x_att)
        x = x * x_att # 使用注意力权重对输入特征图 x 进行加权
        x = self.norm(x) # 使用 Batch Normalization 规范化注意力加权后的特征图

        # 在高度和宽度维度上对特征图进行逐像素采样，实现下采样
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        
        # 将采样结果在通道维度上拼接
        x = torch.cat((
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        # 经过卷积操作，得到下采样后的输出
        return self.conv(x)


# 定义FPNet类，集成了SRM预处理、特征传递块、注意力下采样模块和分类头
class FPNet(nn.Module):
    def __init__(self):
        super(FPNet, self).__init__()
        # SRM预处理
        self.srm = SRMConv2d(1, 0)
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(30)
        # ReLU激活函数
        self.act = nn.ReLU(inplace=True)

        # 特征传递块1~4
        self.fpb1 = FPB(30, 30)
        self.fpb2 = FPB(30*2, 30)
        self.fpb3 = FPB(30, 30)
        self.fpb4 = FPB(30*2, 30)

        # 注意力下采样模块1
        self.adm1 = ADM(30, 64)
        # 特征传递块5
        self.fpb5 = FPB(64, 64)
        # 注意力下采样模块2
        self.adm2 = ADM(64, 128)
        # 特征传递块6
        self.fpb6 = FPB(128, 128)

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout层
        self.dropout = nn.Dropout(p=0.1)
        # 全连接层（分类头）
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        # 将输入转换为float类型
        x = x.float()
        # SRM预处理
        x = self.srm(x)
        # 批归一化
        x = self.bn1(x)
        # ReLU激活
        x = self.act(x)

        # 特征传递块1
        res1 = x
        x = self.fpb1(x)
        x = torch.cat([x, res1], dim=1)
        # 特征传递块2
        x = self.fpb2(x)
        res2 = x
        # 特征传递块3
        x = self.fpb3(x)
        x = torch.cat([x, res2], dim=1)
        # 特征传递块4
        x = self.fpb4(x)

        # 注意力下采样模块1
        x = self.adm1(x)
        # 特征传递块5
        x = self.fpb5(x)

        # 注意力下采样模块2
        x = self.adm2(x)
        # 特征传递块6
        x = self.fpb6(x)

        # 全局平均池化
        out = self.avgpool(x)
        out_flatten = out.view(out.size(0), out.size(1))
        # Dropout
        out_flatten = self.dropout(out_flatten)
        # 全连接层（分类头）
        out = self.fc(out_flatten)
        return out


if __name__ == '__main__':
    from thop import profile

    # 定义输入张量
    x = torch.randn(1, 1, 256, 256)

    # 实例化模型
    net = FPNet()

    # 计算模型flops和params
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)

    # # 使用os.path.join 拼接同目录的images文件夹
    # cover_path = os.path.join(os.path.dirname(__file__), 'images', 'cover')
    # stego_path = os.path.join(os.path.dirname(__file__), 'images', 'stego')

    # # 检测cover/1.pgm,让net判别
    # cover_img = Image.open(f"{cover_path}/1.pgm")
    # cover_img = cover_img.resize((256, 256))
    # cover_img = np.array(cover_img)
    # # cover_img = cover_img.transpose((2, 0, 1))
    # cover_img = cover_img.reshape((1, 1, 256, 256))
    # cover_img = torch.from_numpy(cover_img)
    # cover_img = cover_img.float()

    # # 检测stego/1.pgm,让net判别
    # stego_img = Image.open(f"{stego_path}/1.pgm")
    # stego_img = stego_img.resize((256, 256))
    # stego_img = np.array(stego_img)
    # # stego_img = stego_img.transpose((2, 0, 1))
    # stego_img = stego_img.reshape((1, 1, 256, 256))
    # stego_img = torch.from_numpy(stego_img)
    # stego_img = stego_img.float()

    # # 将图片送入net中
    # out = net(cover_img)
    # out2 = net(cover_img)

    # # 输出结果
    # print(out)
    # print(out2)
    # input("按回车键结束")

    # cover_judge_to_stego = 0
    # cover_judge_to_cover = 0
    # stego_judge_to_stego = 0
    # stego_judge_to_cover = 0
    # for i in range(1,10001):
    #     print("第", i, "张")
    #     cover_img = Image.open(f"{cover_path}/{i}.pgm")
    #     cover_img = cover_img.resize((256, 256))
    #     cover_img = np.array(cover_img)
    #     # cover_img = cover_img.transpose((2, 0, 1))
    #     cover_img = cover_img.reshape((1, 1, 256, 256))
    #     cover_img = torch.from_numpy(cover_img)
    #     cover_img = cover_img.float()

    #     stego_img = Image.open(f"{stego_path}/{i}.pgm")
    #     stego_img = stego_img.resize((256, 256))
    #     stego_img = np.array(stego_img)
    #     # stego_img = stego_img.transpose((2, 0, 1))
    #     stego_img = stego_img.reshape((1, 1, 256, 256))
    #     stego_img = torch.from_numpy(stego_img)
    #     stego_img = stego_img.float()

    #     out = net(cover_img)
    #     out2 = net(stego_img)
    #     print(out)
    #     print(out2)



    #     if out[0][0] < out[0][1]:
    #         print("cover judge as stego")
    #         cover_judge_to_stego += 1
    #     else:
    #         print("cover judge as cover")
    #         cover_judge_to_cover += 1

    #     if out2[0][0] < out2[0][1]:
    #         print("stego judge as stego")
    #         stego_judge_to_stego += 1
    #     else:
    #         print("stego judge as cover")
    #         stego_judge_to_cover += 1

    # print("cover judge to stego: ", cover_judge_to_stego)
    # print("cover judge to cover: ", cover_judge_to_cover)
    # print("stego judge to stego: ", stego_judge_to_stego)
    # print("stego judge to cover: ", stego_judge_to_cover)

