import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from pathlib import Path
from .preprocessing import preprocess_mri, normalize_cfdna_features


class MultiModalDataset(Dataset):
    """
    Enhanced multi-modal dataset for MRI and cfDNA features with advanced data handling
    
    Features:
    - Automatic data validation and cleaning
    - Data augmentation support
    - Memory-efficient loading
    - Statistics tracking
    - Cross-validation splits
    - Data integrity checks
    """
    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        target_shape: Tuple[int, int, int] = (16, 128, 128),
        transform=None,
        cache_data: bool = False,
        normalize_cfdna: bool = True,
        validate_integrity: bool = True,
        min_intensity_threshold: float = 0.01,
        max_missing_ratio: float = 0.1
    ):
        """
        Args:
            data_dir: Root directory containing patient folders
            labels_file: CSV file with patient_id and label columns
            target_shape: Target MRI shape (D, H, W)
            transform: Optional data augmentation
            cache_data: Whether to cache preprocessed data in memory
            normalize_cfdna: Whether to normalize cfDNA features
            validate_integrity: Whether to validate data integrity
            min_intensity_threshold: Minimum intensity threshold for valid MRI
            max_missing_ratio: Maximum allowed missing data ratio
        """
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.transform = transform
        self.cache_data = cache_data
        self.normalize_cfdna = normalize_cfdna
        self.validate_integrity = validate_integrity
        self.min_intensity_threshold = min_intensity_threshold
        self.max_missing_ratio = max_missing_ratio
        
        # Initialize data containers
        self.data_cache = {} if cache_data else None
        self.cfdna_scaler = None
        self.data_stats = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load and validate labels
        self._load_labels(labels_file)
        
        # Validate data exists and integrity
        self._validate_data()
        
        # Initialize cfDNA normalization if requested
        if self.normalize_cfdna:
            self._initialize_cfdna_normalization()
        
        # Calculate dataset statistics
        self._calculate_statistics()
    
    def _load_labels(self, labels_file: str):
        """Load and validate labels file"""
        try:
            self.labels_df = pd.read_csv(labels_file)
            
            # Validate required columns
            required_cols = ['patient_id', 'label']
            missing_cols = [col for col in required_cols if col not in self.labels_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for duplicates
            duplicates = self.labels_df['patient_id'].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate patient IDs, removing...")
                self.labels_df = self.labels_df.drop_duplicates('patient_id')
            
            # Validate labels
            unique_labels = self.labels_df['label'].unique()
            if not set(unique_labels).issubset({0, 1}):
                raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")
            
            self.patient_ids = self.labels_df['patient_id'].tolist()
            self.labels = self.labels_df['label'].tolist()
            
            self.logger.info(f"Loaded {len(self.patient_ids)} patients from labels file")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load labels from {labels_file}: {str(e)}")
    
    def _validate_data(self):
        """Enhanced data validation with integrity checks"""
        valid_patients = []
        valid_labels = []
        validation_issues = []
        
        for idx, patient_id in enumerate(self.patient_ids):
            patient_dir = self.data_dir / patient_id
            
            # Check if patient directory exists
            if not patient_dir.exists():
                validation_issues.append(f"Patient {patient_id}: directory not found")
                continue
            
            # Check required files
            required_files = ['T1.npy', 'T2.npy', 'DWI.npy', 'cfdna_features.npy']
            file_paths = {name: patient_dir / name for name in required_files}
            
            missing_files = [name for name, path in file_paths.items() if not path.exists()]
            if missing_files:
                validation_issues.append(f"Patient {patient_id}: missing files {missing_files}")
                continue
            
            # Validate file integrity if requested
            if self.validate_integrity:
                try:
                    self._validate_patient_data_integrity(patient_id, file_paths)
                except Exception as e:
                    validation_issues.append(f"Patient {patient_id}: integrity check failed - {str(e)}")
                    continue
            
            valid_patients.append(patient_id)
            valid_labels.append(self.labels[idx])
        
        # Update valid data
        self.patient_ids = valid_patients
        self.labels = valid_labels
        
        # Log validation results
        total_patients = len(self.labels_df)
        valid_count = len(valid_patients)
        invalid_count = total_patients - valid_count
        
        self.logger.info(f"Data validation complete:")
        self.logger.info(f"  Valid patients: {valid_count}/{total_patients}")
        self.logger.info(f"  Invalid patients: {invalid_count}")
        
        if validation_issues:
            self.logger.warning(f"Validation issues found:")
            for issue in validation_issues[:10]:  # Show first 10 issues
                self.logger.warning(f"  {issue}")
            if len(validation_issues) > 10:
                self.logger.warning(f"  ... and {len(validation_issues) - 10} more issues")
        
        if valid_count == 0:
            raise RuntimeError("No valid patients found in dataset")
        
        # Check class balance
        pos_ratio = np.mean(valid_labels)
        self.logger.info(f"Class distribution: {pos_ratio:.3f} positive, {1-pos_ratio:.3f} negative")
        
        if pos_ratio < 0.01 or pos_ratio > 0.99:
            self.logger.warning(f"Severe class imbalance detected: {pos_ratio:.3f} positive ratio")
    
    def _validate_patient_data_integrity(self, patient_id: str, file_paths: Dict[str, Path]):
        """Validate individual patient data integrity"""
        # Check MRI data
        mri_shapes = []
        for mri_type in ['T1', 'T2', 'DWI']:
            try:
                data = np.load(file_paths[f'{mri_type}.npy'])
                
                # Check data type and range
                if not np.isfinite(data).all():
                    raise ValueError(f"{mri_type} contains non-finite values")
                
                # Check minimum intensity
                if np.max(data) < self.min_intensity_threshold:
                    raise ValueError(f"{mri_type} maximum intensity too low: {np.max(data)}")
                
                # Check for reasonable shape
                if len(data.shape) < 3 or any(s < 8 for s in data.shape[-3:]):
                    raise ValueError(f"{mri_type} has invalid shape: {data.shape}")
                
                mri_shapes.append(data.shape)
                
            except Exception as e:
                raise ValueError(f"Failed to load {mri_type}: {str(e)}")
        
        # Check cfDNA features
        try:
            cfdna_data = np.load(file_paths['cfdna_features.npy'])
            
            if not np.isfinite(cfdna_data).all():
                raise ValueError("cfDNA features contain non-finite values")
            
            if len(cfdna_data.shape) != 1:
                raise ValueError(f"cfDNA features should be 1D, got shape: {cfdna_data.shape}")
            
            # Check for reasonable feature count
            if len(cfdna_data) < 10 or len(cfdna_data) > 10000:
                self.logger.warning(f"Patient {patient_id}: unusual cfDNA feature count: {len(cfdna_data)}")
            
        except Exception as e:
            raise ValueError(f"Failed to load cfDNA features: {str(e)}")
    
    def _initialize_cfdna_normalization(self):
        """Initialize cfDNA feature normalization"""
        if len(self.patient_ids) == 0:
            return
        
        self.logger.info("Initializing cfDNA normalization...")
        
        # Collect all cfDNA features for normalization
        all_cfdna_features = []
        
        for patient_id in self.patient_ids[:min(100, len(self.patient_ids))]:  # Sample for efficiency
            try:
                patient_dir = self.data_dir / patient_id
                cfdna_path = patient_dir / 'cfdna_features.npy'
                cfdna_features = np.load(cfdna_path)
                all_cfdna_features.append(cfdna_features)
            except Exception as e:
                self.logger.warning(f"Failed to load cfDNA for normalization from {patient_id}: {str(e)}")
        
        if all_cfdna_features:
            all_cfdna_features = np.array(all_cfdna_features)
            self.cfdna_scaler = StandardScaler()
            self.cfdna_scaler.fit(all_cfdna_features)
            self.logger.info(f"cfDNA normalization initialized with {len(all_cfdna_features)} samples")
    
    def _calculate_statistics(self):
        """Calculate dataset statistics"""
        if len(self.patient_ids) == 0:
            return
        
        self.logger.info("Calculating dataset statistics...")
        
        # Basic statistics
        self.data_stats = {
            'total_patients': len(self.patient_ids),
            'positive_samples': sum(self.labels),
            'negative_samples': len(self.labels) - sum(self.labels),
            'positive_ratio': np.mean(self.labels),
            'target_shape': self.target_shape
        }
        
        # Sample a few patients for detailed stats
        sample_size = min(10, len(self.patient_ids))
        sample_indices = np.random.choice(len(self.patient_ids), sample_size, replace=False)
        
        mri_stats = {'T1': [], 'T2': [], 'DWI': []}
        cfdna_dims = []
        
        for idx in sample_indices:
            patient_id = self.patient_ids[idx]
            try:
                patient_dir = self.data_dir / patient_id
                
                # MRI statistics
                for mri_type in ['T1', 'T2', 'DWI']:
                    data = np.load(patient_dir / f'{mri_type}.npy')
                    mri_stats[mri_type].append({
                        'shape': data.shape,
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'min': float(np.min(data)),
                        'max': float(np.max(data))
                    })
                
                # cfDNA statistics
                cfdna_data = np.load(patient_dir / 'cfdna_features.npy')
                cfdna_dims.append(len(cfdna_data))
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate stats for {patient_id}: {str(e)}")
        
        # Store aggregated statistics
        if cfdna_dims:
            self.data_stats['cfdna_dim'] = {
                'mean': float(np.mean(cfdna_dims)),
                'std': float(np.std(cfdna_dims)),
                'min': int(np.min(cfdna_dims)),
                'max': int(np.max(cfdna_dims))
            }
        
        self.logger.info(f"Dataset statistics calculated for {sample_size} samples")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced data loading with caching and error handling
        
        Returns:
            mri_tensor: (3, D, H, W) - T1, T2, DWI
            cfdna_tensor: (cfdna_dim,)
            label_tensor: scalar
        """
        patient_id = self.patient_ids[idx]
        
        # Check cache first
        if self.cache_data and patient_id in self.data_cache:
            mri_data, cfdna_features = self.data_cache[patient_id]
        else:
            try:
                mri_data, cfdna_features = self._load_patient_data(patient_id)
                
                # Cache if requested
                if self.cache_data:
                    self.data_cache[patient_id] = (mri_data, cfdna_features)
                    
            except Exception as e:
                self.logger.error(f"Failed to load data for patient {patient_id}: {str(e)}")
                # Return zero tensors as fallback
                mri_data = np.zeros((3, *self.target_shape), dtype=np.float32)
                cfdna_features = np.zeros(512, dtype=np.float32)  # Default cfDNA dim
        
        # Convert to tensors
        mri_tensor = torch.from_numpy(mri_data).float()
        cfdna_tensor = torch.from_numpy(cfdna_features).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            mri_tensor = self.transform(mri_tensor)
        
        return mri_tensor, cfdna_tensor, label_tensor
    
    def _load_patient_data(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess data for a single patient"""
        patient_dir = self.data_dir / patient_id
        
        # Load MRI data
        t1 = np.load(patient_dir / 'T1.npy')
        t2 = np.load(patient_dir / 'T2.npy')
        dwi = np.load(patient_dir / 'DWI.npy')
        
        # Preprocess MRI with error handling
        try:
            t1 = preprocess_mri(t1, self.target_shape)
            t2 = preprocess_mri(t2, self.target_shape)
            dwi = preprocess_mri(dwi, self.target_shape)
        except Exception as e:
            self.logger.warning(f"MRI preprocessing failed for {patient_id}: {str(e)}")
            # Use zero-filled arrays as fallback
            t1 = np.zeros(self.target_shape, dtype=np.float32)
            t2 = np.zeros(self.target_shape, dtype=np.float32)
            dwi = np.zeros(self.target_shape, dtype=np.float32)
        
        # Stack MRI channels
        mri_data = np.stack([t1, t2, dwi], axis=0)  # (3, D, H, W)
        
        # Load and normalize cfDNA features
        cfdna_features = np.load(patient_dir / 'cfdna_features.npy')
        
        if self.normalize_cfdna and self.cfdna_scaler is not None:
            try:
                cfdna_features = self.cfdna_scaler.transform(cfdna_features.reshape(1, -1)).flatten()
            except Exception as e:
                self.logger.warning(f"cfDNA normalization failed for {patient_id}: {str(e)}")
        
        return mri_data, cfdna_features
    
    def get_patient_info(self, idx: int) -> Dict[str, Union[str, int, float]]:
        """Get detailed information about a patient"""
        patient_id = self.patient_ids[idx]
        patient_dir = self.data_dir / patient_id
        
        info = {
            'patient_id': patient_id,
            'label': self.labels[idx],
            'index': idx
        }
        
        try:
            # Add file sizes
            for file_name in ['T1.npy', 'T2.npy', 'DWI.npy', 'cfdna_features.npy']:
                file_path = patient_dir / file_name
                if file_path.exists():
                    info[f'{file_name}_size_mb'] = file_path.stat().st_size / (1024 * 1024)
            
            # Add data shapes
            for mri_type in ['T1', 'T2', 'DWI']:
                try:
                    data = np.load(patient_dir / f'{mri_type}.npy')
                    info[f'{mri_type}_shape'] = data.shape
                    info[f'{mri_type}_dtype'] = str(data.dtype)
                except:
                    pass
            
            # Add cfDNA info
            try:
                cfdna_data = np.load(patient_dir / 'cfdna_features.npy')
                info['cfdna_dim'] = len(cfdna_data)
                info['cfdna_dtype'] = str(cfdna_data.dtype)
                info['cfdna_mean'] = float(np.mean(cfdna_data))
                info['cfdna_std'] = float(np.std(cfdna_data))
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"Failed to get info for patient {patient_id}: {str(e)}")
        
        return info
    
    def create_train_val_split(self, val_ratio: float = 0.2, random_state: int = 42) -> Tuple['MultiModalDataset', 'MultiModalDataset']:
        """Create stratified train/validation split"""
        train_indices, val_indices = train_test_split(
            range(len(self)), 
            test_size=val_ratio, 
            random_state=random_state,
            stratify=self.labels
        )
        
        # Create subset datasets
        train_dataset = DatasetSubset(self, train_indices)
        val_dataset = DatasetSubset(self, val_indices)
        
        self.logger.info(f"Created train/val split: {len(train_indices)}/{len(val_indices)}")
        
        return train_dataset, val_dataset
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            return torch.tensor([1.0, 1.0])
        
        total = len(self.labels)
        pos_weight = total / (2.0 * pos_count)
        neg_weight = total / (2.0 * neg_count)
        
        return torch.tensor([neg_weight, pos_weight])
    
    def clear_cache(self):
        """Clear data cache to free memory"""
        if self.data_cache:
            self.data_cache.clear()
            self.logger.info("Data cache cleared")
    
    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """Get dataset statistics"""
        return self.data_stats.copy()
    
    def print_summary(self):
        """Print dataset summary"""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total patients: {len(self.patient_ids)}")
        print(f"Positive samples: {sum(self.labels)} ({np.mean(self.labels):.3f})")
        print(f"Negative samples: {len(self.labels) - sum(self.labels)} ({1-np.mean(self.labels):.3f})")
        print(f"Target MRI shape: {self.target_shape}")
        print(f"Data directory: {self.data_dir}")
        print(f"Cache enabled: {self.cache_data}")
        print(f"cfDNA normalization: {self.normalize_cfdna}")
        print(f"Data validation: {self.validate_integrity}")
        
        if self.data_stats.get('cfdna_dim'):
            cfdna_stats = self.data_stats['cfdna_dim']
            print(f"cfDNA dimensions: {cfdna_stats['min']}-{cfdna_stats['max']} (avg: {cfdna_stats['mean']:.1f})")
        
        print("="*60)


class DatasetSubset(Dataset):
    """Dataset subset for train/val splits"""
    def __init__(self, dataset: MultiModalDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    @property
    def labels(self):
        return [self.dataset.labels[i] for i in self.indices]
    
    @property
    def patient_ids(self):
        return [self.dataset.patient_ids[i] for i in self.indices]