import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion_position, criterion_force, optimizer, device):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    running_accuracy_direction = 0.0
    running_accuracy_position = 0.0
    running_mse_force = 0.0
    running_mae_force = 0.0
    total_batches = len(dataloader)

    # 遍历数据集的每一个批次
    for i, data in enumerate(dataloader, 0):
        inputs, labels_direction, labels_position, labels_force = data
        inputs, labels_direction, labels_position, labels_force = inputs.to(device), labels_direction.to(device), labels_position.to(device), labels_force.to(device)
        # 重置梯度
        optimizer.zero_grad()
        # 前向传播
        outputs_direction, outputs_position, outputs_force = model(inputs)
        # 计算损失
        loss_direction = criterion_position(outputs_direction, labels_direction)
        loss_position = criterion_position(outputs_position, labels_position)
        loss_force = criterion_force(outputs_force.squeeze(), labels_force)
        loss = 1.0 * loss_direction + 1.0 * loss_position + 2.0 * loss_force
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 计算总损失
        running_loss += loss.item()

        # 计算准确率 和 mse
        predicted_direction = torch.max(outputs_direction, 1)[1].cpu().numpy()
        predicted_position = torch.max(outputs_position, 1)[1].cpu().numpy()
        outputs_force = outputs_force.squeeze().detach().cpu().numpy()
        labels_direction = labels_direction.cpu().numpy()
        labels_position = labels_position.cpu().numpy()
        labels_force = labels_force.cpu().numpy()
        accuracy_direction = accuracy_score(labels_direction, predicted_direction)
        accuracy_position = accuracy_score(labels_position, predicted_position)
        mse_force = mean_squared_error(outputs_force, labels_force)
        mae_force = mean_absolute_error(outputs_force, labels_force)

        running_accuracy_direction += accuracy_direction
        running_accuracy_position += accuracy_position
        running_mse_force += mse_force
        running_mae_force += mae_force

    # 输出整个 epoch 的平均损失
    epoch_loss = running_loss / total_batches
    epoch_accuracy_direction = running_accuracy_direction / total_batches
    epoch_accuracy_position = running_accuracy_position / total_batches
    epoch_mse_force = running_mse_force / total_batches
    epoch_mae_force = running_mae_force / total_batches
    tqdm.write(f'Epoch Loss: {epoch_loss:.10f}, Accuracy Direction: {epoch_accuracy_direction:.4f}, Accuracy Position: {epoch_accuracy_position:.4f}, MSE Force: {epoch_mse_force:.4f}, MAE Force: {epoch_mae_force:.4f}')
    return epoch_loss, epoch_accuracy_direction, epoch_accuracy_position, epoch_mse_force, epoch_mae_force
