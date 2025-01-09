import os
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from models import FBGNet , MultiTaskTransformer, ResNet1D, PatchTST, CONFIGS
from config import MODEL_SAVE_DIR, NUM_EPOCHS
from train import train_one_epoch
from test import test_one_epoch
from datas.data_preprocess import get_dataloaders
from datas.valid_true_preprocess import get_valid_dataloaders
from utils.graph_test_model import valid_true_evaluate


def init_wandb():
    wandb.init(
        project="Fibre_Optical",
        entity="chengjun_team",
        config={
            "num_epochs": NUM_EPOCHS,
            "checkpoint_path": MODEL_SAVE_DIR,
            "learning_rate": 0.001,
            "architecture": "分类＋回归",
            "dataset": "data.txt"
        }
    )


def train_model(use_wandb=False, num_epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_wandb:
        init_wandb()
    
    # 创建数据集实例
    train_dataloader, test_dataloader = get_dataloaders(batch_size=196)

    # valid_dataloader = get_valid_dataloaders(batch_size=60)
    # model = PatchTST(num_classes_1=25, num_classes_2=24, configs=CONFIGS)
    
    model = PatchTST(num_classes_1=25, num_classes_2=24, configs=CONFIGS)
    model.to(device)
    torch.compile(model)

    # 损失函数和优化器
    criterion_position = nn.CrossEntropyLoss()
    criterion_force = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    decay_rate = 0.2
    decay_steps = [70, 140]

    best_mae = float('inf')  # Initialize best MAE as infinity
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        # 学习率衰减
        if epoch in decay_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate

        # 训练和测试
        train_results = train_one_epoch(model, train_dataloader, criterion_position, criterion_force, optimizer, device)
        train_loss, train_accuracy_direction, train_accuracy_position, train_mse_force, train_mae_force = train_results
        
        test_results = test_one_epoch(model, test_dataloader, criterion_position, criterion_force, device)
        test_loss, test_accuracy_direction, test_accuracy_position, test_mse_force, test_mae_force = test_results

        # Save model if we get better MAE
        if test_mae_force < best_mae:
            best_mae = test_mae_force
            if not os.path.exists(MODEL_SAVE_DIR):
                os.makedirs(MODEL_SAVE_DIR)
            model_path = os.path.join(MODEL_SAVE_DIR, "full_data_best_model.pth")
            print(f"New best MAE: {best_mae:.4f}, saving model to {model_path}")
            torch.save(model.state_dict(), model_path)

        # valid_results = valid_true_evaluate(model, valid_dataloader, device)
        # direction_cm, position_cm = valid_results['direction_cm'], valid_results['position_cm']
        # valid_accuracy_direction, valid_accuracy_position, valid_mse_force = valid_results['direction_accuracy'], valid_results['position_accuracy'], valid_results['force_mse']


        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "test_loss": test_loss,
                #########
                "train_accuracy_direction": train_accuracy_direction,
                "train_accuracy_position": train_accuracy_position,
                "train_mse_force": train_mse_force,
                "train_mae_force": train_mae_force,
                ######
                "test_accuracy_direction": test_accuracy_direction,
                "test_accuracy_position": test_accuracy_position,
                "test_mse_force": test_mse_force,
                "test_mae_force": test_mae_force,
                # #########
                # "valid_accuracy_direction": valid_accuracy_direction,
                # "valid_accuracy_position": valid_accuracy_position,
                # "valid_mse_force": valid_mse_force,
                # #########
                # "direction_cm": wandb.Image(direction_cm),
                # "position_cm": wandb.Image(position_cm)
            })

    # # 保存模型
    # if not os.path.exists(MODEL_SAVE_DIR):
    #     os.makedirs(MODEL_SAVE_DIR)
    # model_path = os.path.join(MODEL_SAVE_DIR, "model.pth")
    # print(f"Saving model to {model_path}")
    # torch.save(model.state_dict(), model_path)
    
    # # 测试模型
    # test_model(test_dataloader, model, model_path)

    if use_wandb:
        wandb.finish()
    


def main():
    train_model(use_wandb=True, num_epochs=200)

if __name__ == "__main__":
    main()