"""
Simplified visualizer module for CNN+LSTM+Attention model training
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Optional, Dict, Any
import torch

def plot_losses(train_losses: List[float], val_losses: List[float], save_path: str = None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_losses_logscale(train_losses: List[float], val_losses: List[float], save_path: str = None):
    """Plot losses in log scale"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label='Train Loss', alpha=0.8)
    plt.semilogy(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Validation Losses (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_lr(lr_history: List[float], save_path: str = None):
    """Plot learning rate schedule"""
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_grad_norm(grad_norms: List[float], save_path: str = None):
    """Plot gradient norms"""
    plt.figure(figsize=(10, 6))
    plt.plot(grad_norms, alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm During Training')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_param_count(param_counts: Dict[str, int], save_path: str = None):
    """Plot parameter counts"""
    plt.figure(figsize=(10, 6))
    names = list(param_counts.keys())
    counts = list(param_counts.values())
    plt.bar(names, counts)
    plt.xlabel('Layer')
    plt.ylabel('Parameter Count')
    plt.title('Model Parameter Distribution')
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(targets: np.ndarray, predictions: np.ndarray, save_path: str = None):
    """Plot predictions vs targets"""
    plt.figure(figsize=(12, 8))
    
    # Time series plot
    plt.subplot(2, 1, 1)
    time_steps = range(min(500, len(targets)))  # Show first 500 points
    plt.plot(time_steps, targets[:len(time_steps)], 'k-', label='Ground Truth', alpha=0.8)
    plt.plot(time_steps, predictions[:len(time_steps)], 'r-', label='Predictions', alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Predictions vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(2, 1, 2)
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.6, s=20)
    min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Prediction Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residual_hist(residuals: np.ndarray, save_path: str = None):
    """Plot residual histogram"""
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_interval(targets: np.ndarray, predictions: np.ndarray, 
                           lower_bound: np.ndarray = None, upper_bound: np.ndarray = None, 
                           save_path: str = None):
    """Plot predictions with confidence intervals"""
    plt.figure(figsize=(12, 6))
    time_steps = range(min(200, len(targets)))
    
    plt.plot(time_steps, targets[:len(time_steps)], 'k-', label='Ground Truth', alpha=0.8)
    plt.plot(time_steps, predictions[:len(time_steps)], 'r-', label='Predictions', alpha=0.8)
    
    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(time_steps, lower_bound[:len(time_steps)], 
                        upper_bound[:len(time_steps)], alpha=0.3, label='Confidence Interval')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Predictions with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_multihorizon_error(errors: Dict[int, List[float]], save_path: str = None):
    """Plot errors for different prediction horizons"""
    plt.figure(figsize=(10, 6))
    
    for horizon, error_list in errors.items():
        plt.plot(error_list, label=f'Horizon {horizon}', alpha=0.8)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.title('Multi-horizon Prediction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Placeholder functions for advanced visualizations
def plot_attention_heatmap(attention_weights: torch.Tensor, save_path: str = None):
    """Plot attention heatmap (placeholder)"""
    pass

def plot_attention_multihead(attention_weights: torch.Tensor, save_path: str = None):
    """Plot multi-head attention (placeholder)"""
    pass

def plot_lstm_hidden_heatmap(hidden_states: torch.Tensor, save_path: str = None):
    """Plot LSTM hidden state heatmap (placeholder)"""
    pass

def plot_cnn_feature_maps(feature_maps: torch.Tensor, save_path: str = None):
    """Plot CNN feature maps (placeholder)"""
    pass

def plot_series_distribution(data: np.ndarray, save_path: str = None):
    """Plot time series distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=50, alpha=0.7, density=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Data Distribution')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_corr_heatmap(correlation_matrix: np.ndarray, feature_names: List[str] = None, save_path: str = None):
    """Plot correlation heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Heatmap')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_split_distribution(train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray = None, save_path: str = None):
    """Plot data split distributions"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(train_data.flatten(), bins=50, alpha=0.7, label='Train', density=True)
    plt.hist(val_data.flatten(), bins=50, alpha=0.7, label='Validation', density=True)
    if test_data is not None:
        plt.hist(test_data.flatten(), bins=50, alpha=0.7, label='Test', density=True)
    
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Data Split Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_performance(performance_over_time: List[float], save_path: str = None):
    """Plot temporal performance metrics"""
    plt.figure(figsize=(10, 6))
    plt.plot(performance_over_time, alpha=0.8)
    plt.xlabel('Time Window')
    plt.ylabel('Performance Metric')
    plt.title('Temporal Performance Analysis')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_all_plots(outputs: Dict[str, Any], save_dir: str):
    """Save all plots to directory"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    if 'train_losses' in outputs and 'val_losses' in outputs:
        plot_losses(outputs['train_losses'], outputs['val_losses'], 
                   os.path.join(save_dir, 'losses.png'))
    
    if 'predictions' in outputs and 'targets' in outputs:
        plot_predictions(outputs['targets'], outputs['predictions'],
                        os.path.join(save_dir, 'predictions.png'))
    
    print(f"Plots saved to {save_dir}")