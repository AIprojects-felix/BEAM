"""Serialize training-fitted cfDNA scaling with the model checkpoint."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def scaler_to_state(scaler):
    check_is_fitted(scaler)
    return {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_features': int(scaler.n_features_in_),
        'n_samples': int(scaler.n_samples_seen_),
    }


def scaler_from_checkpoint(checkpoint, expected_dim):
    state = checkpoint.get('cfdna_scaler')
    if state is None:
        raise ValueError(
            'Checkpoint has no training-fitted cfDNA scaler. Use a checkpoint '
            'with its original training preprocessing parameters; never fit '
            'a replacement scaler on evaluation or prediction patients.'
        )
    if state['n_features'] != expected_dim:
        raise ValueError('Checkpoint scaler dimension does not match model input')
    scaler = StandardScaler()
    for key, attr in [('mean', 'mean_'), ('scale', 'scale_'), ('var', 'var_')]:
        values = np.asarray(state[key], dtype=np.float64)
        if values.shape != (expected_dim,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid checkpoint scaler {key}')
        setattr(scaler, attr, values)
    if np.any(scaler.scale_ <= 0) or np.any(scaler.var_ < 0):
        raise ValueError('Invalid checkpoint scaler variance or scale')
    scaler.n_features_in_ = expected_dim
    scaler.n_samples_seen_ = int(state['n_samples'])
    return scaler


def transform_cfdna(values, scaler):
    values = np.asarray(values, dtype=np.float64)
    check_is_fitted(scaler)
    if values.shape != (scaler.n_features_in_,) or not np.isfinite(values).all():
        raise ValueError('cfDNA vector has invalid shape or non-finite values')
    return scaler.transform(values.reshape(1, -1))[0].astype(np.float32)
