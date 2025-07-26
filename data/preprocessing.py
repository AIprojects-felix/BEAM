import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler


def preprocess_mri(mri_data: np.ndarray, target_shape: tuple = (16, 128, 128)) -> np.ndarray:
    """
    Preprocess MRI data
    
    Args:
        mri_data: Input MRI volume
        target_shape: Target shape (D, H, W)
    
    Returns:
        Preprocessed MRI data
    """
    # Handle 4D data
    if len(mri_data.shape) == 4:
        mri_data = mri_data[0]
    
    # Resize to target shape
    if mri_data.shape != target_shape:
        zoom_factors = [t/s for t, s in zip(target_shape, mri_data.shape)]
        mri_data = ndimage.zoom(mri_data, zoom_factors, order=1)
    
    # Intensity normalization
    p1, p99 = np.percentile(mri_data, [0.5, 99.5])
    mri_data = np.clip(mri_data, p1, p99)
    
    if p99 > p1:
        mri_data = (mri_data - p1) / (p99 - p1)
    
    # Standardization
    mean, std = mri_data.mean(), mri_data.std()
    if std > 0:
        mri_data = (mri_data - mean) / std
    
    return mri_data.astype(np.float32)


def normalize_cfdna_features(features: np.ndarray, scaler: StandardScaler = None, fit: bool = True):
    """
    Normalize cfDNA features
    
    Args:
        features: cfDNA features array
        scaler: StandardScaler instance
        fit: Whether to fit the scaler
    
    Returns:
        Normalized features and scaler
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        features_normalized = scaler.fit_transform(features)
    else:
        features_normalized = scaler.transform(features)
    
    return features_normalized, scaler


class DataAugmentation:
    """
    Data augmentation for multi-modal medical data
    
    Features:
    - MRI-specific augmentations
    - cfDNA feature augmentations
    - Consistency preservation across modalities
    """
    
    def __init__(self, 
                 mri_augmentation: bool = True,
                 cfdna_augmentation: bool = True,
                 preserve_consistency: bool = True):
        self.mri_augmentation = mri_augmentation
        self.cfdna_augmentation = cfdna_augmentation
        self.preserve_consistency = preserve_consistency
    
    def augment_mri(self, mri_data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Apply MRI-specific augmentations"""
        if seed is not None:
            np.random.seed(seed)
        
        augmented = mri_data.copy()
        
        # Random rotation (small angles)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-5, 5)
            augmented = self._rotate_3d(augmented, angle)
        
        # Random intensity scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            augmented = augmented * scale
        
        # Random noise addition
        if np.random.random() < 0.3:
            noise_std = 0.01 * np.std(augmented)
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented = augmented + noise
        
        # Random elastic deformation (simplified)
        if np.random.random() < 0.2:
            augmented = self._elastic_deform_3d(augmented)
        
        return augmented
    
    def augment_cfdna(self, cfdna_features: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Apply cfDNA-specific augmentations"""
        if seed is not None:
            np.random.seed(seed)
        
        augmented = cfdna_features.copy()
        
        # Random feature dropout
        if np.random.random() < 0.3:
            dropout_ratio = np.random.uniform(0.01, 0.05)
            mask = np.random.random(augmented.shape) > dropout_ratio
            augmented = augmented * mask
        
        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            augmented = augmented * scale
        
        # Random noise
        if np.random.random() < 0.4:
            noise_std = 0.01 * np.std(augmented)
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented = augmented + noise
        
        return augmented
    
    def _rotate_3d(self, volume: np.ndarray, angle: float) -> np.ndarray:
        """Apply 3D rotation to volume"""
        # Simplified rotation around z-axis for each slice
        rotated = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            # Convert angle to radians
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Simple rotation matrix application
            slice_data = volume[i]
            center = np.array(slice_data.shape) // 2
            
            # This is a simplified rotation - in practice, use scipy.ndimage.rotate
            rotated[i] = ndimage.rotate(slice_data, angle, reshape=False, mode='reflect')
        
        return rotated
    
    def _elastic_deform_3d(self, volume: np.ndarray, alpha: float = 10, sigma: float = 3) -> np.ndarray:
        """Apply elastic deformation to 3D volume"""
        # Generate random displacement fields
        shape = volume.shape
        
        # Create displacement fields
        dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        dz = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        
        # Create coordinate grids
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
        
        # Apply displacements
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
        
        # Interpolate
        deformed = ndimage.map_coordinates(volume, indices, order=1, mode='reflect')
        deformed = deformed.reshape(shape)
        
        return deformed