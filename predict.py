"""
BEAM inference: single-patient or batch prediction. Outputs cancer probability.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.preprocessing import preprocess_mri
from models import BEAM
from utils import create_dirs, load_config


def load_patient(pdir: Path, target_shape):
    """Load the four .npy files inside one patient directory."""
    pdir = Path(pdir)
    t1 = preprocess_mri(np.load(pdir / 'T1.npy'), target_shape)
    t2 = preprocess_mri(np.load(pdir / 'T2.npy'), target_shape)
    dwi = preprocess_mri(np.load(pdir / 'DWI.npy'), target_shape)
    mri = np.stack([t1, t2, dwi], axis=0)              # (3, D, H, W)
    cfdna = np.load(pdir / 'cfdna_features.npy').astype(np.float32)
    return mri, cfdna


@torch.no_grad()
def infer(model, mri: np.ndarray, cfdna: np.ndarray, device) -> float:
    mri_t = torch.from_numpy(mri).float().unsqueeze(0).to(device)
    cfd_t = torch.from_numpy(cfdna).float().unsqueeze(0).to(device)
    logit = model(mri_t, cfd_t)
    return float(torch.sigmoid(logit).item())


def main(args):
    config = load_config(args.config) if args.config else {}
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    cfdna_dim = args.cfdna_dim if args.cfdna_dim else model_cfg.get('cfdna_dim', 275)
    d_model = model_cfg.get('d_model', 512)
    num_heads = model_cfg.get('num_heads', 8)
    num_layers = model_cfg.get('num_layers', 3)
    dropout = model_cfg.get('dropout', 0.1)
    target_shape = tuple(data_cfg.get('mri_shape', [16, 128, 128]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BEAM(cfdna_dim=cfdna_dim, d_model=d_model, num_heads=num_heads,
                 num_layers=num_layers, dropout=dropout).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {args.model_path}')

    create_dirs([args.output_dir])

    if args.batch_mode:
        if not args.data_dir:
            raise ValueError('Batch mode requires --data_dir')
        data_root = Path(args.data_dir)
        patient_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
        rows = []
        for pdir in patient_dirs:
            try:
                mri, cfdna = load_patient(pdir, target_shape)
            except FileNotFoundError:
                print(f'Skipping {pdir.name}: required files missing')
                continue
            prob = infer(model, mri, cfdna, device)
            rows.append({
                'patient_id': pdir.name,
                'cancer_probability': prob,
                'predicted_label': int(prob > args.threshold),
            })
        df = pd.DataFrame(rows)
        out_csv = os.path.join(args.output_dir, 'batch_predictions.csv')
        df.to_csv(out_csv, index=False)
        print(f'Saved {len(df)} predictions to {out_csv}')
    else:
        if not args.mri_dir:
            raise ValueError('Single-patient mode requires --mri_dir')
        mri, cfdna = load_patient(args.mri_dir, target_shape)
        prob = infer(model, mri, cfdna, device)
        pid = args.patient_id or Path(args.mri_dir).name
        label = 'cancer' if prob > args.threshold else 'non-cancer'
        print(f'\nPatient {pid}: cancer probability = {prob:.4f} '
              f'(threshold {args.threshold} -> prediction = {label})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BEAM inference')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output/pred')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cfdna_dim', type=int, default=0)

    parser.add_argument('--batch_mode', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='batch mode: root directory containing patient subdirectories')

    parser.add_argument('--mri_dir', type=str, default=None,
                        help='single-patient mode: directory with T1.npy/T2.npy/DWI.npy/cfdna_features.npy')
    parser.add_argument('--patient_id', type=str, default=None)

    main(parser.parse_args())
