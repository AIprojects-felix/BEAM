"""
Generate a tiny dummy dataset used by the pipeline check.

The volumes and feature vectors are randomly drawn and intended only to verify
that the codebase loads, trains, evaluates, and predicts end-to-end. Positive
and negative samples come from slightly different distributions so that the
pipeline check produces non-trivial training behaviour.

Usage:
    python data/make_sample_data.py --output_dir data/sample --n_pos 10 --n_neg 10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_patient(out_dir: Path, label: int, rng: np.random.Generator,
                 mri_shape=(16, 128, 128), cfdna_dim=275):
    out_dir.mkdir(parents=True, exist_ok=True)

    # MRI: positive samples receive a slightly brighter central blob
    for name in ['T1', 'T2', 'DWI']:
        vol = rng.normal(0.3, 0.1, mri_shape).astype(np.float32)
        vol = np.clip(vol, 0, 1)
        if label == 1:
            d, h, w = mri_shape
            cz, cy, cx = d // 2, h // 2, w // 2
            zs, ys, xs = slice(cz - 2, cz + 2), slice(cy - 16, cy + 16), slice(cx - 16, cx + 16)
            vol[zs, ys, xs] += rng.normal(0.4, 0.1, vol[zs, ys, xs].shape).astype(np.float32)
            vol = np.clip(vol, 0, 1)
        np.save(out_dir / f'{name}.npy', vol)

    # cfDNA: positive vs negative differ in mean
    if label == 1:
        cfdna = rng.normal(0.3, 1.0, cfdna_dim).astype(np.float32)
    else:
        cfdna = rng.normal(-0.3, 1.0, cfdna_dim).astype(np.float32)
    np.save(out_dir / 'cfdna_features.npy', cfdna)


def main(args):
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    pid = 1
    for label, n in [(1, args.n_pos), (0, args.n_neg)]:
        for _ in range(n):
            patient_id = f'patient_{pid:03d}'
            make_patient(out_root / patient_id, label, rng,
                         mri_shape=tuple(args.mri_shape), cfdna_dim=args.cfdna_dim)
            rows.append({'patient_id': patient_id, 'label': label})
            pid += 1

    pd.DataFrame(rows).to_csv(out_root / 'labels.csv', index=False)
    print(f'Generated {len(rows)} dummy patients in {out_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/sample')
    parser.add_argument('--n_pos', type=int, default=10)
    parser.add_argument('--n_neg', type=int, default=10)
    parser.add_argument('--cfdna_dim', type=int, default=275)
    parser.add_argument('--mri_shape', type=int, nargs=3, default=[16, 128, 128])
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
