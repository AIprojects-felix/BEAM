"""MRI preprocessing."""

import numpy as np
from scipy import ndimage


def preprocess_mri(mri_data: np.ndarray, target_shape: tuple = (16, 128, 128)) -> np.ndarray:
    """
    Per-sequence MRI preprocessing:

    1. If the input is 4D, take the first volume.
    2. Resample to target_shape with linear interpolation if shapes differ.
    3. Clip intensities to the [0.5, 99.5] percentile range.
    4. Linearly rescale to [0, 1].

    Isotropic resampling to 1.0 x 1.0 x 1.0 mm^3 is expected to have been
    performed during ROI cropping; this function assumes the input volume is
    already gland-cropped and brings it to (16, 128, 128).
    """
    if mri_data.ndim == 4:
        mri_data = mri_data[0]

    if mri_data.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, mri_data.shape)]
        mri_data = ndimage.zoom(mri_data, zoom_factors, order=1)

    p_low, p_high = np.percentile(mri_data, [0.5, 99.5])
    mri_data = np.clip(mri_data, p_low, p_high)

    if p_high > p_low:
        mri_data = (mri_data - p_low) / (p_high - p_low)
    else:
        mri_data = np.zeros_like(mri_data)

    return mri_data.astype(np.float32)
