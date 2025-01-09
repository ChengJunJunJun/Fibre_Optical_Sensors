import torch
import numpy as np
from torch.utils.data import DataLoader
from .FBG_Dataset import FBGDataset

def get_valid_dataloaders(batch_size=60):
    """
    Load and preprocess validation/test data from external test files.
    
    Args:
        batch_size (int): Batch size for the dataloader. Defaults to 60.
        
    Returns:
        DataLoader: DataLoader for validation/test data
    """
    # Load test data
    x_data = np.loadtxt('Data_sets/test_data.txt', delimiter=',')
    y = np.loadtxt('Data_sets/test_label.txt', delimiter=',')
    
    # Reshape and transpose data
    x_data = x_data.reshape(60, 2, 2000)
    normalized_data_x = np.transpose(x_data, (0, 2, 1))
    
    # Convert to PyTorch tensors
    x_tensor = torch.from_numpy(normalized_data_x).float()
    y_direction_tensor = torch.from_numpy(y[:, 0]).long()
    y_position_tensor = torch.from_numpy(y[:, 1]).long()
    y_force_tensor = torch.from_numpy(y[:, 2]).float()
    
    # Create dataset and dataloader
    valid_dataset = FBGDataset(
        x_tensor, 
        y_direction_tensor, 
        y_position_tensor, 
        y_force_tensor, 
        train=False
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return valid_dataloader

if __name__ == '__main__':
    valid_dataloader = get_valid_dataloaders()
    print(len(valid_dataloader))