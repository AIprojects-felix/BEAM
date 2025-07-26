import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': recall_score(1-y_true, 1-y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                   save_path: Optional[str] = None) -> Tuple[float, plt.Figure]:
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    
    Returns:
        AUC score and figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return auc, fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    
    Returns:
        Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig