import argparse
import os
import time
import json
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import wandb
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

from models import BEAM
from data import MultiModalDataset
from utils import (
    set_seed, save_checkpoint, load_checkpoint, get_logger, create_dirs, load_config,
    calculate_metrics, plot_roc_curve, plot_confusion_matrix
)

warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, clip_grad=1.0):
    """Enhanced training epoch with mixed precision and gradient clipping"""
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (mri_data, cfdna_data, labels) in enumerate(progress_bar):
        mri_data = mri_data.to(device, non_blocking=True)
        cfdna_data = cfdna_data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(mri_data, cfdna_data)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(mri_data, cfdna_data)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        with torch.no_grad():
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_probs)
    )
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device, return_predictions=False):
    """Enhanced validation with optional prediction return"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    all_patient_ids = []
    
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_data in progress_bar:
            if len(batch_data) == 4:  # If patient IDs are included
                mri_data, cfdna_data, labels, patient_ids = batch_data
                all_patient_ids.extend(patient_ids)
            else:
                mri_data, cfdna_data, labels = batch_data
            
            mri_data = mri_data.to(device, non_blocking=True)
            cfdna_data = cfdna_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(mri_data, cfdna_data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)
    probs_array = np.array(all_probs)
    
    metrics = calculate_metrics(labels_array, preds_array, probs_array)
    
    if return_predictions:
        predictions_df = pd.DataFrame({
            'patient_id': all_patient_ids if all_patient_ids else range(len(labels_array)),
            'true_label': labels_array,
            'predicted_label': preds_array,
            'cancer_probability': probs_array
        })
        return avg_loss, metrics, predictions_df
    
    return avg_loss, metrics, labels_array, probs_array


def create_weighted_sampler(labels):
    """Create weighted sampler for handling class imbalance"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def setup_scheduler(optimizer, scheduler_type, **kwargs):
    """Setup learning rate scheduler"""
    if scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 30), gamma=kwargs.get('gamma', 0.1))
    else:
        return None


def log_training_progress(epoch, train_metrics, val_metrics, lr, runtime, logger, wandb_run=None):
    """Log training progress to console, file, and wandb"""
    logger.info(f"\nEpoch {epoch} Summary:")
    logger.info(f"  Training   - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    logger.info(f"  Validation - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    logger.info(f"  Learning Rate: {lr:.2e}, Runtime: {runtime:.1f}s")
    
    if wandb_run:
        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_accuracy': val_metrics['accuracy'],
            'learning_rate': lr,
            'epoch_time': runtime
        })


def cross_validate(args, k_folds=5):
    """Perform k-fold cross validation"""
    set_seed(args.seed)
    
    dataset = MultiModalDataset(args.data_path, args.labels_file)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    
    cv_results = defaultdict(list)
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(range(len(dataset)), dataset.labels)):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("-" * 30)
        
        # Further split train_val into train and val
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, random_state=args.seed,
            stratify=[dataset.labels[i] for i in train_val_idx]
        )
        
        # Create data loaders for this fold
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Train model for this fold
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BEAM(
            cfdna_dim=args.cfdna_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        
        # Quick training (reduced epochs for CV)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0
        for epoch in range(min(args.epochs, 20)):  # Reduced epochs for CV
            train_epoch(model, train_loader, optimizer, criterion, device)
            _, val_metrics, _, _ = validate(model, val_loader, criterion, device)
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
        
        # Test on fold
        _, test_metrics, _, _ = validate(model, test_loader, criterion, device)
        
        # Store results
        for metric, value in test_metrics.items():
            cv_results[metric].append(value)
    
    # Calculate mean and std for each metric
    print("\nCross-Validation Results:")
    print("=" * 40)
    for metric, values in cv_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    return cv_results


def plot_training_history(train_history, val_history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(train_history['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(val_history['loss'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[0, 1].plot(train_history['auc'], label='Train', linewidth=2)
    axes[0, 1].plot(val_history['auc'], label='Validation', linewidth=2)
    axes[0, 1].set_title('AUC', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(train_history['accuracy'], label='Train', linewidth=2)
    axes[1, 0].plot(val_history['accuracy'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(train_history['f1'], label='Train', linewidth=2)
    axes[1, 1].plot(val_history['f1'], label='Validation', linewidth=2)
    axes[1, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    start_time = time.time()
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"exp_{timestamp}")
    create_dirs([exp_dir, os.path.join(exp_dir, 'checkpoints'), os.path.join(exp_dir, 'plots')])
    
    # Logger
    logger = get_logger('BEAM', os.path.join(exp_dir, 'train.log'))
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Save configuration
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            project="BEAM-ProstateCancer",
            config=vars(args),
            name=f"beam_{timestamp}"
        )
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Cross-validation if requested
    if args.cross_validate:
        cv_results = cross_validate(args)
        return cv_results
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = MultiModalDataset(args.data_path, args.labels_file)
    
    # Calculate dataset statistics
    labels = np.array(dataset.labels)
    pos_ratio = np.mean(labels)
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Positive ratio: {pos_ratio:.3f} ({np.sum(labels)}/{len(labels)})")
    
    # Split dataset
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, 
        stratify=labels
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=args.val_size, random_state=args.seed,
        stratify=labels[train_indices]
    )
    
    logger.info(f"Data split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Handle class imbalance
    train_labels = labels[train_indices]
    if args.use_weighted_sampler:
        sampler = create_weighted_sampler(train_labels)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        shuffle=shuffle, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model initialization
    logger.info("Initializing model...")
    model = BEAM(
        cfdna_dim=args.cfdna_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function setup
    if args.use_focal_loss:
        pos_weight = torch.tensor([len(train_labels) / np.sum(train_labels) - 1]).to(device)
        criterion = FocalLoss(alpha=pos_weight.item(), gamma=2.0)
        logger.info(f"Using Focal Loss with alpha={pos_weight.item():.3f}")
    elif args.use_class_weights:
        pos_weight = torch.tensor([len(train_labels) / np.sum(train_labels) - 1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Using weighted BCE Loss with pos_weight={pos_weight.item():.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Using standard BCE Loss")
    
    # Optimizer setup
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay,
            momentum=0.9
        )
    
    # Learning rate scheduler
    scheduler = setup_scheduler(optimizer, args.scheduler, T_max=args.epochs, step_size=args.epochs//3)
    
    # Mixed precision training
    scaler = GradScaler() if args.use_amp else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001) if args.early_stopping else None
    
    # Resume training if checkpoint exists
    start_epoch = 0
    best_val_auc = 0.0
    train_history = defaultdict(list)
    val_history = defaultdict(list)
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming training from {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint['best_metric']
    
    # Training loop
    logger.info("Starting training...")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, args.clip_grad
        )
        
        # Validation phase
        val_loss, val_metrics, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])
            else:
                scheduler.step()
        
        # Record metrics
        train_metrics['loss'] = train_loss
        val_metrics['loss'] = val_loss
        
        for key, value in train_metrics.items():
            train_history[key].append(value)
        for key, value in val_metrics.items():
            val_history[key].append(value)
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        log_training_progress(epoch + 1, train_metrics, val_metrics, current_lr, epoch_time, logger, wandb_run)
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            save_checkpoint(
                model, optimizer, epoch, best_val_auc,
                os.path.join(exp_dir, 'checkpoints', 'best_model.pth'),
                scheduler=scheduler.state_dict() if scheduler else None,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                train_history=train_history,
                val_history=val_history
            )
            logger.info(f"  ✓ New best model saved with AUC: {best_val_auc:.4f}")
            
            # Plot and save ROC curve
            _, roc_fig = plot_roc_curve(val_labels, val_probs, 
                                      os.path.join(exp_dir, 'plots', 'best_val_roc.png'))
            if wandb_run:
                wandb.log({"val_roc_curve": wandb.Image(roc_fig)})
            plt.close(roc_fig)
        
        # Regular checkpoint saving
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['auc'],
                os.path.join(exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'),
                scheduler=scheduler.state_dict() if scheduler else None,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )
        
        # Early stopping check
        if early_stopping and early_stopping(val_metrics['auc'], model):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Training completed
    total_training_time = time.time() - training_start_time
    logger.info(f"\nTraining completed in {total_training_time:.1f} seconds")
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Load best model for final evaluation
    load_checkpoint(os.path.join(exp_dir, 'checkpoints', 'best_model.pth'), model, device=device)
    
    # Final test evaluation
    logger.info("\nFinal evaluation on test set...")
    test_loss, test_metrics, test_predictions = validate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    logger.info("\nFinal Test Results:")
    logger.info("=" * 50)
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Save test predictions
    test_predictions.to_csv(os.path.join(exp_dir, 'test_predictions.csv'), index=False)
    
    # Save final results
    final_results = {
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'total_training_time': total_training_time,
        'total_epochs': epoch + 1 if 'epoch' in locals() else args.epochs
    }
    
    with open(os.path.join(exp_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Plot final visualizations
    test_labels = test_predictions['true_label'].values
    test_probs = test_predictions['cancer_probability'].values
    test_preds = test_predictions['predicted_label'].values
    
    # ROC curve
    plot_roc_curve(test_labels, test_probs, os.path.join(exp_dir, 'plots', 'final_test_roc.png'))
    
    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, os.path.join(exp_dir, 'plots', 'final_test_confusion.png'))
    
    # Training history plots
    plot_training_history(train_history, val_history, os.path.join(exp_dir, 'plots', 'training_history.png'))
    
    if wandb_run:
        wandb.log(final_results)
        wandb.finish()
    
    logger.info(f"\nAll results saved to: {exp_dir}")
    
    return final_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced BEAM model training with advanced features')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to labels CSV file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.125,
                        help='Validation set ratio from train+val')
    
    # Model arguments
    parser.add_argument('--cfdna_dim', type=int, default=512,
                        help='cfDNA feature dimension')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    
    # Loss function options
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss for class imbalance')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights in BCE loss')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Use weighted sampler for training data')
    
    # Advanced training features
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Checkpoint saving frequency (epochs)')
    
    # Experiment management
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='Output directory')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    # Cross-validation
    parser.add_argument('--cross_validate', action='store_true',
                        help='Perform k-fold cross validation instead of single training')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross validation')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle scheduler 'none' case
    if args.scheduler == 'none':
        args.scheduler = None
    
    # Validation checks
    if args.use_focal_loss and args.use_class_weights:
        print("Warning: Both focal loss and class weights specified. Using focal loss.")
        args.use_class_weights = False
    
    if args.cross_validate:
        print(f"Running {args.k_folds}-fold cross validation...")
        cv_results = cross_validate(args, args.k_folds)
    else:
        print("Running single training...")
        results = main(args)