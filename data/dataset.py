"""Multi-modal dataset: T1 / T2 / DWI volumes + cfDNA feature vector + label."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

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

        # cfDNA standardization (fit on the full dataset)
        self.cfdna_scaler = None
        if self.normalize_cfdna:
            cfdna_all = np.stack([
                np.load(self.data_dir / pid / 'cfdna_features.npy')
                for pid in self.patient_ids
            ])
            self.cfdna_scaler = StandardScaler()
            self.cfdna_scaler.fit(cfdna_all)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pid = self.patient_ids[idx]
        pdir = self.data_dir / pid

        t1 = preprocess_mri(np.load(pdir / 'T1.npy'), self.target_shape)
        t2 = preprocess_mri(np.load(pdir / 'T2.npy'), self.target_shape)
        dwi = preprocess_mri(np.load(pdir / 'DWI.npy'), self.target_shape)
        mri = np.stack([t1, t2, dwi], axis=0)        # (3, D, H, W)

        cfdna = np.load(pdir / 'cfdna_features.npy').astype(np.float32)
        if self.cfdna_scaler is not None:
            cfdna = self.cfdna_scaler.transform(cfdna.reshape(1, -1)).flatten().astype(np.float32)

        return (
            torch.from_numpy(mri).float(),
            torch.from_numpy(cfdna).float(),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
