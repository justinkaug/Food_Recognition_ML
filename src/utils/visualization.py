"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from PIL import Image


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def show_batch(images, labels, class_names=None, predictions=None, n=16):
    """Display a batch of images with labels"""
    n = min(n, len(images))
    rows = int(np.sqrt(n))
    cols = (n + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if n > 1 else [axes]
    
    for idx in range(n):
        ax = axes[idx]
        
        # Denormalize image
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Create title
        label = labels[idx].item()
        title = class_names[label] if class_names else f'Label: {label}'
        
        if predictions is not None:
            pred = predictions[idx].item()
            pred_name = class_names[pred] if class_names else f'Pred: {pred}'
            color = 'green' if pred == label else 'red'
            title = f'True: {title}\nPred: {pred_name}'
            ax.set_title(title, color=color)
        else:
            ax.set_title(title)
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels, class_names=None, save_path=None):
    """Plot class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique)), counts)
    
    if class_names:
        plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    else:
        plt.xticks(range(len(unique)), unique)
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(model, dataloader, class_names, device='cuda', n=16):
    """Visualize model predictions"""
    model.eval()
    
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = outputs.max(1)
    
    show_batch(images, labels, class_names, predictions, n)


def plot_feature_maps(model, image, layer_name, save_path=None):
    """Visualize feature maps from a specific layer"""
    # This is a placeholder - actual implementation would require hooks
    pass


def plot_grad_cam(model, image, target_class, save_path=None):
    """Generate and plot Grad-CAM visualization"""
    # This is a placeholder - actual implementation would require Grad-CAM library
    pass
