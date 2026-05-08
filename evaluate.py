"""
BEAM evaluation: run a locked model on a labelled cohort and report AUROC,
accuracy, sensitivity, specificity, Brier score, etc.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import MultiModalDataset
from models import BEAM
from utils import (
    calculate_metrics, create_dirs, load_config,
    plot_confusion_matrix, plot_roc_curve,
)


def main(args):
    config = load_config(args.config) if args.config else {}
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    cfdna_dim   = args.cfdna_dim   if args.cfdna_dim   else model_cfg.get('cfdna_dim', 275)
    d_model     = model_cfg.get('d_model', 512)
    num_heads   = model_cfg.get('num_heads', 8)
    num_layers  = model_cfg.get('num_layers', 3)
    dropout     = model_cfg.get('dropout', 0.1)
    target_shape = tuple(data_cfg.get('mri_shape', [16, 128, 128]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    create_dirs([args.output_dir])

    # Data
    dataset = MultiModalDataset(args.test_data, args.labels_file, target_shape=target_shape)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = BEAM(cfdna_dim=cfdna_dim, d_model=d_model, num_heads=num_heads,
                 num_layers=num_layers, dropout=dropout).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {args.model_path}')

    # Inference
    all_probs, all_labels = [], []
    with torch.no_grad():
        for mri, cfdna, labels in tqdm(loader, desc='Evaluate', leave=False):
            mri = mri.to(device); cfdna = cfdna.to(device)
            logits = model(mri, cfdna)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > args.threshold).astype(int)
    metrics = calculate_metrics(labels, preds, probs)

    print('\nEvaluation metrics:')
    for k, v in metrics.items():
        print(f'  {k:12s} = {v:.4f}')

    # Save outputs
    pd.DataFrame({
        'patient_id': dataset.patient_ids,
        'true_label': labels.astype(int),
        'predicted_label': preds,
        'cancer_probability': probs,
    }).to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_roc_curve(labels, probs, os.path.join(args.output_dir, 'roc.png'))
    plot_confusion_matrix(labels, preds, os.path.join(args.output_dir, 'confusion.png'))

    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BEAM evaluation')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True,
                        help='root directory containing one subdirectory per patient')
    parser.add_argument('--labels_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output/eval')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cfdna_dim', type=int, default=0)
    main(parser.parse_args())
