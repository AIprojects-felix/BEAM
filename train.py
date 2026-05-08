"""
BEAM training script.

- AdamW (lr=1e-4, weight_decay=1e-4)
- BCE loss
- 100 epochs, batch size = 4
- 70 / 10 / 20 stratified split into train / tuning / hold-out test
- Locked checkpoint = the one with the highest tuning-set AUROC.
"""

import argparse
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import MultiModalDataset
from models import BEAM
from utils import (
    calculate_metrics, create_dirs, get_logger, load_config,
    plot_confusion_matrix, plot_roc_curve, save_checkpoint, set_seed,
)


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    losses, all_probs, all_labels = [], [], []
    desc = 'Train' if is_train else 'Eval'

    with torch.set_grad_enabled(is_train):
        for mri, cfdna, labels in tqdm(loader, desc=desc, leave=False):
            mri = mri.to(device, non_blocking=True)
            cfdna = cfdna.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(mri, cfdna)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)
    metrics = calculate_metrics(labels, preds, probs)
    metrics['loss'] = float(np.mean(losses))
    return metrics, labels, probs


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train', lw=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation', lw=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Training Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_auc'], label='Train', lw=2)
    axes[1].plot(epochs, history['val_auc'], label='Validation', lw=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUROC')
    axes[1].set_title('Validation AUROC'); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    base = save_path.rsplit('.', 1)[0]
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)


def main(args):
    config = load_config(args.config) if args.config else {}
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    # CLI args take precedence only when explicitly provided (non-zero default)
    cfdna_dim   = args.cfdna_dim   if args.cfdna_dim   else model_cfg.get('cfdna_dim', 275)
    d_model     = model_cfg.get('d_model', 512)
    num_heads   = model_cfg.get('num_heads', 8)
    num_layers  = model_cfg.get('num_layers', 3)
    dropout     = model_cfg.get('dropout', 0.1)

    epochs       = args.epochs       if args.epochs       else train_cfg.get('epochs', 100)
    batch_size   = args.batch_size   if args.batch_size   else train_cfg.get('batch_size', 4)
    lr           = train_cfg.get('learning_rate', 1e-4)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    num_workers  = train_cfg.get('num_workers', 4)
    test_size    = train_cfg.get('test_size', 0.2)
    val_size     = train_cfg.get('val_size', 0.125)
    target_shape = tuple(data_cfg.get('mri_shape', [16, 128, 128]))
    seed         = config.get('seed', args.seed)

    set_seed(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.output_dir, f'exp_{timestamp}')
    create_dirs([exp_dir, os.path.join(exp_dir, 'checkpoints'), os.path.join(exp_dir, 'plots')])
    logger = get_logger('BEAM', os.path.join(exp_dir, 'train.log'))
    logger.info(f'Experiment directory: {exp_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Dataset
    logger.info('Loading dataset...')
    dataset = MultiModalDataset(args.data_path, args.labels_file, target_shape=target_shape)
    labels = np.asarray(dataset.labels)
    logger.info(f'N = {len(dataset)}, positive ratio = {labels.mean():.3f}')

    # 70 / 10 / 20 stratified split
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=test_size,
                                           random_state=seed, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size,
                                          random_state=seed, stratify=labels[train_idx])
    logger.info(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = BEAM(
        cfdna_dim=cfdna_dim, d_model=d_model,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Trainable parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    best_val_auc = -1.0
    best_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')

    logger.info('Starting training...')
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_m, _, _ = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_m, val_labels, val_probs = run_one_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['train_auc'].append(train_m['auc'])
        history['val_auc'].append(val_m['auc'])

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_m['loss']:.4f} train_auc={train_m['auc']:.4f} | "
            f"val_loss={val_m['loss']:.4f} val_auc={val_m['auc']:.4f}"
        )

        # Locked checkpoint = highest validation AUROC
        if val_m['auc'] > best_val_auc:
            best_val_auc = val_m['auc']
            save_checkpoint(model, optimizer, epoch, best_val_auc, best_path,
                            train_metrics=train_m, val_metrics=val_m)
            logger.info(f'  -> saved new best (val AUROC = {best_val_auc:.4f})')

    logger.info(f'Training done in {time.time() - t0:.1f}s, best val AUROC = {best_val_auc:.4f}')

    # Training curves
    plot_history(history, os.path.join(exp_dir, 'plots', 'training_history.png'))

    # Evaluate the best (locked) model on the hold-out test set
    logger.info('Evaluating best model on hold-out test set...')
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_m, test_labels, test_probs = run_one_epoch(model, test_loader, criterion, device)
    test_preds = (test_probs > 0.5).astype(int)
    logger.info('Hold-out test metrics:')
    for k, v in test_m.items():
        logger.info(f'  {k} = {v:.4f}')

    plot_roc_curve(test_labels, test_probs, os.path.join(exp_dir, 'plots', 'test_roc.png'))
    plot_confusion_matrix(test_labels, test_preds,
                          os.path.join(exp_dir, 'plots', 'test_confusion.png'))

    with open(os.path.join(exp_dir, 'final_results.json'), 'w') as f:
        json.dump({
            'best_val_auc': best_val_auc,
            'test_metrics': test_m,
            'history': history,
            'config': {
                'cfdna_dim': cfdna_dim, 'd_model': d_model, 'num_heads': num_heads,
                'num_layers': num_layers, 'dropout': dropout,
                'epochs': epochs, 'batch_size': batch_size,
                'learning_rate': lr, 'weight_decay': weight_decay,
                'seed': seed,
            },
        }, f, indent=2)

    logger.info(f'Results saved to {exp_dir}')
    return exp_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BEAM training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--labels_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--epochs', type=int, default=0, help='override config.training.epochs')
    parser.add_argument('--batch_size', type=int, default=0, help='override config.training.batch_size')
    parser.add_argument('--cfdna_dim', type=int, default=0, help='override config.model.cfdna_dim')
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
