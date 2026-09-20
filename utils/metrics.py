"""Evaluation metrics and basic plotting."""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# Global matplotlib style: English labels in Times New Roman with larger fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Compute AUROC, accuracy, sensitivity, specificity, precision, F1,
    and Brier score."""
    has_two_classes = len(np.unique(y_true)) > 1
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)) if has_two_classes else 0.0,
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'sensitivity': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(recall_score(1 - y_true, 1 - y_pred, zero_division=0)),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'brier':       float(brier_score_loss(y_true, y_prob)) if has_two_classes else 0.0,
    }


def _save_fig(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save the figure as both PNG (300 dpi) and PDF (vector)."""
    if save_path is None:
        return
    base = save_path.rsplit('.', 1)[0]
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None
) -> Tuple[float, plt.Figure]:
    """ROC curve with AUC in the legend."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    _save_fig(fig, save_path)
    return auc, fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
) -> plt.Figure:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=ax,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    _save_fig(fig, save_path)
    return fig
