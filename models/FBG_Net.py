import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 模型架构
# 定义一个卷积神经网络，全连接神经网络，和任务学习，输出力的位置（分类任务）和大小（回归任务）
# 分类任务利用交叉熵损失函数，回归任务利用均方误差损失函数

class FBGNet(nn.Module):
    def __init__(self):
        super(FBGNet, self).__init__()
        # 一维卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 497, 128)  # 64个通道，每个通道497个特征点
        # 输出层
        self.fc_position = nn.Linear(128, 24)  # 用于分类任务：24个位置
        self.fc_force = nn.Linear(128, 1)  # 用于回归任务：力的大小

    def forward(self, x):
        # 输入x的形状为 (batch_size, 2000)，需要reshape为 (batch_size, 1, 2000)
        x = x.unsqueeze(1)
        # 卷积层 + ReLU + 池化层
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 32, 998)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 64, 497)
        # 展平
        x = x.view(-1, 64 * 497)  # (batch_size, 64*497)
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))  # (batch_size, 128)

        # 输出位置分类和力大小回归
        position_output = self.fc_position(x)  # (batch_size, 24)

        force_output = self.fc_force(x)  # (batch_size, 1)
        return position_output, force_output

if __name__ == '__main__':
    model = FBGNet()

    print(model)
    from torchinfo import summary
    summary(model, input_size=(10, 2000))  # do a test pass through of an example input size