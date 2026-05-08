"""
Interface and dimensionality layout for the five cfDNA feature modules.

The cfDNA branch of BEAM expects, per patient, a single 275-dimensional
feature vector obtained by concatenating five domain-specific modules:

    CNV      (39  dims)  Chromosomal-arm-level z-scores from 1-Mb CNV bins.
    FSR      (30  dims)  Fragment Size Ratio (5-Mb bins) after GC correction + PCA.
    Griffin  (100 dims)  Nucleosome-positioning coverage features over a panel
                         of 50 prostate-relevant transcription factors
                         (2 metrics per TF).
    MutCS    (85  dims)  COSMIC SBS signature contributions retained after
                         filtering by mean activity.
    FragMA   (21  dims)  cfDNA 5'-end-motif features.

Total:                   275 dims.

The wet-lab feature-extraction pipeline (alignment, CNV calling, fragmentomic
profiling, etc.) runs upstream of this repository. The functions below
formalize the per-module signatures and the concatenation order so that the
upstream pipeline produces a vector compatible with ``MultiModalDataset``.
"""

from typing import Dict
import numpy as np


# ---------- Per-module feature dimensions ----------
CNV_DIM     = 39
FSR_DIM     = 30
GRIFFIN_DIM = 100
MUTCS_DIM   = 85
FRAGMA_DIM  = 21

CFDNA_DIM = CNV_DIM + FSR_DIM + GRIFFIN_DIM + MUTCS_DIM + FRAGMA_DIM
assert CFDNA_DIM == 275, f"Module dimensions must sum to 275, got {CFDNA_DIM}"


def _not_provided(name: str) -> None:
    raise NotImplementedError(
        f"{name} is produced by the upstream cfDNA pipeline. "
        "Provide the module output via the wet-lab feature-extraction stack, "
        "and store the concatenated 275-dim vector as cfdna_features.npy."
    )


def extract_cnv(bam_path: str) -> np.ndarray:
    """Chromosomal-arm-level z-scores from 1-Mb CNV bins."""
    _not_provided("extract_cnv")


def extract_fsr(bam_path: str) -> np.ndarray:
    """Fragment Size Ratio with GC correction and PCA reduction."""
    _not_provided("extract_fsr")


def extract_griffin(bam_path: str) -> np.ndarray:
    """Griffin coverage features over a panel of 50 prostate-relevant transcription factors."""
    _not_provided("extract_griffin")


def extract_mutcs(bam_path: str) -> np.ndarray:
    """Retained COSMIC SBS signature contributions."""
    _not_provided("extract_mutcs")


def extract_fragma(bam_path: str) -> np.ndarray:
    """cfDNA 5'-end-motif features."""
    _not_provided("extract_fragma")


def build_cfdna_feature_vector(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate the five module vectors (in the canonical order) into a single
    275-dim input vector.

    Args:
        features: {'cnv': (39,), 'fsr': (30,), 'griffin': (100,), 'mutcs': (85,), 'fragma': (21,)}

    Returns:
        np.ndarray of shape (275,)
    """
    expected = {
        'cnv':     CNV_DIM,
        'fsr':     FSR_DIM,
        'griffin': GRIFFIN_DIM,
        'mutcs':   MUTCS_DIM,
        'fragma':  FRAGMA_DIM,
    }
    parts = []
    for name, dim in expected.items():
        if name not in features:
            raise KeyError(f"Missing module '{name}'")
        v = np.asarray(features[name]).flatten()
        if v.shape[0] != dim:
            raise ValueError(f"Module '{name}' must have dim {dim}, got {v.shape[0]}")
        parts.append(v)
    return np.concatenate(parts).astype(np.float32)
