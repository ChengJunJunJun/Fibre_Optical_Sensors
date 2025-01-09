from torch.utils.data import DataLoader
from .FBG_Dataset import FBGDataset
import torch
import os
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split


def get_dataloaders(batch_size=196):
    # 加载数据
    x_data = np.loadtxt('./Data_sets/data.txt', delimiter=',')
    y = np.loadtxt('./Data_sets/label.txt', delimiter=',')
    
    # 数据预处理
    # 重塑数组并调整轴的顺序
    x_data = x_data.reshape(-1, 2, 2000)
    normalized_data_x = np.transpose(x_data, (0, 2, 1))

    
    # 转换为PyTorch张量
    x_tensor = torch.from_numpy(normalized_data_x).float()
    y_direction_tensor = torch.from_numpy(y[:, 0]).long()
    y_position_tensor = torch.from_numpy(y[:, 1]).long()
    y_force_tensor = torch.from_numpy(y[:, 2]).float()
    
    # 训练集和测试集分割
    x_train, x_test, y_direction_train, y_direction_test, y_position_train, y_position_test, y_force_train, y_force_test = train_test_split(
        x_tensor, y_direction_tensor, y_position_tensor, y_force_tensor, 
        test_size=0.2, random_state=42
    )
    
    # 创建数据集实例
    train_dataset = FBGDataset(x_train, y_direction_train, y_position_train, y_force_train, train=True)
    test_dataset = FBGDataset(x_test, y_direction_test, y_position_test, y_force_test, train=False)
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader
   
if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloaders()
    print(len(train_dataloader))
    print(len(test_dataloader))
