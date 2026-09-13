"""Multi-modal dataset: T1 / T2 / DWI volumes + cfDNA feature vector + label."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from .cfdna_scaling import transform_cfdna

from .preprocessing import preprocess_mri


class MultiModalDataset(Dataset):
    """
    Each patient directory must contain:
        T1.npy, T2.npy, DWI.npy, cfdna_features.npy
    The labels CSV must contain two columns: patient_id, label (0/1).
    """

    DEFAULT_CFDNA_DIM = 275

    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        target_shape: Tuple[int, int, int] = (16, 128, 128),
        normalize_cfdna: bool = True,
        cfdna_scaler=None,
    ):
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.normalize_cfdna = normalize_cfdna

        labels_df = pd.read_csv(labels_file)
        if not {'patient_id', 'label'}.issubset(labels_df.columns):
            raise ValueError("Labels CSV must contain 'patient_id' and 'label' columns")

        # Keep only patients whose files actually exist on disk
        valid = []
        for _, row in labels_df.iterrows():
            pdir = self.data_dir / str(row['patient_id'])
            required = ['T1.npy', 'T2.npy', 'DWI.npy', 'cfdna_features.npy']
            if pdir.exists() and all((pdir / f).exists() for f in required):
                valid.append((str(row['patient_id']), int(row['label'])))

        if not valid:
            raise RuntimeError(f"No complete patient data found under {data_dir}")

        self.patient_ids = [p for p, _ in valid]
        self.labels = [l for _, l in valid]

        # Construction never learns parameters from this cohort.
        self.cfdna_scaler = cfdna_scaler

    def fit_cfdna_scaler(self, train_indices):
        """Call only after splitting, with training indices exclusively."""
        if self.cfdna_scaler is not None:
            raise ValueError('cfDNA scaler is already fitted; refusing to refit')
        indices = list(train_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Training indices must be non-empty and unique')
        if any(i < 0 or i >= len(self) for i in indices):
            raise ValueError('Training index out of range')
        values = np.stack([
            np.load(self.data_dir / self.patient_ids[i] / 'cfdna_features.npy')
            for i in indices
        ])
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('Invalid training cfDNA features')
        self.cfdna_scaler = StandardScaler().fit(values)
        return self.cfdna_scaler

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pid = self.patient_ids[idx]
        pdir = self.data_dir / pid

        t1 = preprocess_mri(np.load(pdir / 'T1.npy'), self.target_shape)
        t2 = preprocess_mri(np.load(pdir / 'T2.npy'), self.target_shape)
        dwi = preprocess_mri(np.load(pdir / 'DWI.npy'), self.target_shape)
        mri = np.stack([t1, t2, dwi], axis=0)        # (3, D, H, W)

        cfdna = np.load(pdir / 'cfdna_features.npy')
        if self.normalize_cfdna:
            if self.cfdna_scaler is None:
                raise ValueError('Fit on training indices or supply the checkpoint scaler first')
            cfdna = transform_cfdna(cfdna, self.cfdna_scaler)

        return (
            torch.from_numpy(mri).float(),
            torch.from_numpy(cfdna).float(),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
