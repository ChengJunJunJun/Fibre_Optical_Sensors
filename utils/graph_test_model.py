import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def plot_confusion_matrix(y_true, y_pred, title):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(cm.shape[1]), yticklabels=range(cm.shape[0]))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    return fig

def valid_true_evaluate(model, valid_loader, device):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics storage
    all_direction_true = []
    all_direction_pred = []
    all_position_true = []
    all_position_pred = []
    all_force_true = []
    all_force_pred = []
    
    with torch.no_grad():
        for inputs, label_direction, labels_position, labels_force in valid_loader:
            # Move inputs to device
            inputs = inputs.to(device)
            
            # Forward pass
            direction_output, position_output, force_output = model(inputs)
            
            # Store predictions and true labels
            all_direction_true.extend(label_direction.cpu().numpy())
            all_direction_pred.extend(torch.argmax(direction_output, dim=1).cpu().numpy())
            
            all_position_true.extend((labels_position - 1).cpu().numpy())
            all_position_pred.extend(torch.argmax(position_output, dim=1).cpu().numpy())
            
            all_force_true.extend(labels_force.cpu().numpy())
            all_force_pred.extend(force_output.cpu().numpy())
    
    # Convert to numpy arrays for easier processing
    all_direction_true = np.array(all_direction_true)
    all_direction_pred = np.array(all_direction_pred)
    all_position_true = np.array(all_position_true)
    all_position_pred = np.array(all_position_pred)
    all_force_true = np.array(all_force_true)
    all_force_pred = np.array(all_force_pred)
    
    # Calculate accuracy for classification tasks
    direction_accuracy = (all_direction_true == all_direction_pred).mean()
    position_accuracy = (all_position_true == all_position_pred).mean()
    
    # Calculate MSE for force regression
    force_mse = np.mean((all_force_true - all_force_pred) ** 2)
    
    # Create confusion matrices
    direction_cm = plot_confusion_matrix(
        torch.tensor(all_direction_true), 
        torch.tensor(all_direction_pred), 
        'Direction Confusion Matrix'
    )
    position_cm = plot_confusion_matrix(
        torch.tensor(all_position_true), 
        torch.tensor(all_position_pred), 
        'Position Confusion Matrix'
    )
    tqdm.write(f'valid_true_evaluate: Direction Accuracy: {direction_accuracy:.4f}, Position Accuracy: {position_accuracy:.4f}, Force MSE: {force_mse:.4f}')
    
    return {
        "direction_accuracy": direction_accuracy,
        "position_accuracy": position_accuracy,
        "force_mse": force_mse,
        
        "predictions": {
            "direction": all_direction_pred,
            "position": all_position_pred,
            "force": all_force_pred
        },
        "direction_cm": direction_cm,
        "position_cm": position_cm
    }
