import argparse
import os
import json
import time
import warnings
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from models import BEAM
from data.preprocessing import preprocess_mri
from data import MultiModalDataset
from torch.utils.data import DataLoader
from utils import load_checkpoint, get_logger, create_dirs

warnings.filterwarnings('ignore')


class BEAMPredictor:
    """Advanced BEAM model predictor with comprehensive analysis"""
    
    def __init__(self, model_path: str, device: torch.device, threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.model = None
        self.model_info = None
        
    def load_model(self, cfdna_dim: int, d_model: int = 512, num_heads: int = 8, 
                   num_layers: int = 3) -> Dict:
        """Load trained BEAM model"""
        self.model = BEAM(
            cfdna_dim=cfdna_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0.0
        ).to(self.device)
        
        checkpoint = load_checkpoint(self.model_path, self.model, device=self.device)
        
        self.model_info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'best_metric': checkpoint.get('best_metric', 'Unknown'),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'model_path': self.model_path
        }
        
        return self.model_info
    
    def load_patient_data(self, mri_dir: str, cfdna_path: str, 
                         target_shape: Tuple[int, int, int] = (16, 128, 128)) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess single patient data"""
        # Validate paths
        if not os.path.exists(mri_dir):
            raise FileNotFoundError(f"MRI directory not found: {mri_dir}")
        if not os.path.exists(cfdna_path):
            raise FileNotFoundError(f"cfDNA file not found: {cfdna_path}")
        
        # Load MRI data
        t1_path = os.path.join(mri_dir, 'T1.npy')
        t2_path = os.path.join(mri_dir, 'T2.npy')
        dwi_path = os.path.join(mri_dir, 'DWI.npy')
        
        missing_files = []
        for name, path in [('T1.npy', t1_path), ('T2.npy', t2_path), ('DWI.npy', dwi_path)]:
            if not os.path.exists(path):
                missing_files.append(name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing MRI files: {', '.join(missing_files)}")
        
        try:
            t1 = preprocess_mri(np.load(t1_path), target_shape)
            t2 = preprocess_mri(np.load(t2_path), target_shape)
            dwi = preprocess_mri(np.load(dwi_path), target_shape)
            
            # Stack MRI channels
            mri_data = np.stack([t1, t2, dwi], axis=0)  # (3, D, H, W)
            
            # Load cfDNA features
            cfdna_features = np.load(cfdna_path)
            
            # Validate data shapes
            if mri_data.shape != (3, *target_shape):
                raise ValueError(f"Invalid MRI shape: {mri_data.shape}, expected: (3, {target_shape})")
            
            return mri_data, cfdna_features
            
        except Exception as e:
            raise RuntimeError(f"Error loading patient data: {str(e)}")
    
    def predict_single(self, mri_data: np.ndarray, cfdna_features: np.ndarray, 
                      return_features: bool = False) -> Dict:
        """Make prediction for single patient with detailed analysis"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Convert to tensors and add batch dimension
        mri_tensor = torch.from_numpy(mri_data).float().unsqueeze(0).to(self.device)
        cfdna_tensor = torch.from_numpy(cfdna_features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            output = self.model(mri_tensor, cfdna_tensor)
            inference_time = time.time() - start_time
            
            probability = torch.sigmoid(output).cpu().item()
            prediction = int(probability > self.threshold)
        
        # Calculate confidence metrics
        confidence = abs(probability - 0.5) * 2  # Distance from decision boundary
        risk_category = self._categorize_risk(probability)
        
        result = {
            'probability': probability,
            'prediction': prediction,
            'risk_category': risk_category,
            'confidence': confidence,
            'threshold_used': self.threshold,
            'inference_time_ms': inference_time * 1000,
            'data_shapes': {
                'mri': mri_data.shape,
                'cfdna': cfdna_features.shape
            }
        }
        
        if return_features:
            # Extract intermediate features if needed
            result['raw_output'] = output.cpu().item()
        
        return result
    
    def predict_batch(self, data_dir: str, patient_list: List[str] = None, 
                     save_results: bool = True, output_dir: str = None) -> pd.DataFrame:
        """Predict for multiple patients in batch"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if patient_list is None:
            # Auto-discover patients
            patient_list = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))]
        
        results = []
        failed_patients = []
        
        print(f"Processing {len(patient_list)} patients...")
        
        for i, patient_id in enumerate(patient_list):
            try:
                patient_dir = os.path.join(data_dir, patient_id)
                mri_dir = patient_dir
                cfdna_path = os.path.join(patient_dir, 'cfdna_features.npy')
                
                # Load data
                mri_data, cfdna_features = self.load_patient_data(mri_dir, cfdna_path)
                
                # Predict
                result = self.predict_single(mri_data, cfdna_features)
                result['patient_id'] = patient_id
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(patient_list)} patients")
                    
            except Exception as e:
                failed_patients.append({'patient_id': patient_id, 'error': str(e)})
                print(f"Failed to process {patient_id}: {str(e)}")
        
        # Create results DataFrame
        if results:
            results_df = pd.DataFrame(results)
        else:
            results_df = pd.DataFrame()
        
        # Save results if requested
        if save_results and output_dir and not results_df.empty:
            results_df.to_csv(os.path.join(output_dir, 'batch_predictions.csv'), index=False)
            
            # Save failed patients log
            if failed_patients:
                failed_df = pd.DataFrame(failed_patients)
                failed_df.to_csv(os.path.join(output_dir, 'failed_patients.csv'), index=False)
        
        print(f"\nBatch prediction completed:")
        print(f"  Successful: {len(results)}")
        print(f"  Failed: {len(failed_patients)}")
        
        return results_df
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk based on probability"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Intermediate"
        else:
            return "High"
    
    def analyze_predictions(self, results_df: pd.DataFrame, save_path: str = None) -> Dict:
        """Analyze batch prediction results"""
        if results_df.empty:
            return {}
        
        analysis = {
            'summary_statistics': {
                'total_patients': len(results_df),
                'predicted_positive': int(results_df['prediction'].sum()),
                'predicted_negative': int(len(results_df) - results_df['prediction'].sum()),
                'mean_probability': float(results_df['probability'].mean()),
                'std_probability': float(results_df['probability'].std()),
                'mean_confidence': float(results_df['confidence'].mean())
            },
            'risk_distribution': results_df['risk_category'].value_counts().to_dict(),
            'probability_percentiles': {
                '25th': float(results_df['probability'].quantile(0.25)),
                '50th': float(results_df['probability'].quantile(0.50)),
                '75th': float(results_df['probability'].quantile(0.75)),
                '95th': float(results_df['probability'].quantile(0.95))
            }
        }
        
        # Generate visualization if save path provided
        if save_path:
            self._plot_prediction_analysis(results_df, save_path)
        
        return analysis
    
    def _plot_prediction_analysis(self, results_df: pd.DataFrame, save_path: str):
        """Generate prediction analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Probability distribution
        axes[0, 0].hist(results_df['probability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.threshold, color='red', linestyle='--', label=f'Threshold: {self.threshold}')
        axes[0, 0].set_xlabel('Cancer Probability')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Probability Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Risk category distribution
        risk_counts = results_df['risk_category'].value_counts()
        axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Risk Category Distribution')
        
        # Confidence vs Probability
        scatter = axes[1, 0].scatter(results_df['probability'], results_df['confidence'], 
                                   c=results_df['prediction'], cmap='RdYlBu', alpha=0.6)
        axes[1, 0].set_xlabel('Cancer Probability')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].set_title('Confidence vs Probability')
        plt.colorbar(scatter, ax=axes[1, 0], label='Prediction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prediction counts
        pred_counts = results_df['prediction'].value_counts()
        axes[1, 1].bar(['Negative', 'Positive'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
                      color=['lightblue', 'lightcoral'])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main(args):
    start_time = time.time()
    
    # Create output directory if needed
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"prediction_{timestamp}")
        create_dirs([output_dir])
    else:
        output_dir = None
    
    # Setup logging
    if output_dir:
        logger = get_logger('BEAM-Predict', os.path.join(output_dir, 'prediction.log'))
    else:
        logger = get_logger('BEAM-Predict')
    
    logger.info(f"Starting BEAM prediction...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Initialize predictor
    predictor = BEAMPredictor(args.model_path, device, args.threshold)
    
    # Load model
    logger.info("Loading model...")
    model_info = predictor.load_model(
        cfdna_dim=args.cfdna_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    logger.info(f"Model loaded successfully:")
    logger.info(f"  Epoch: {model_info['epoch']}")
    logger.info(f"  Best metric: {model_info['best_metric']}")
    logger.info(f"  Parameters: {model_info['total_params']:,}")
    
    # Determine prediction mode
    if args.batch_mode:
        # Batch prediction mode
        logger.info(f"Running batch prediction on directory: {args.data_dir}")
        
        patient_list = None
        if args.patient_list:
            with open(args.patient_list, 'r') as f:
                patient_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Using patient list with {len(patient_list)} patients")
        
        # Run batch prediction
        results_df = predictor.predict_batch(
            data_dir=args.data_dir,
            patient_list=patient_list,
            save_results=True,
            output_dir=output_dir
        )
        
        if not results_df.empty:
            # Analyze results
            analysis = predictor.analyze_predictions(
                results_df, 
                os.path.join(output_dir, 'prediction_analysis.png') if output_dir else None
            )
            
            # Log summary
            logger.info("\nBatch Prediction Summary:")
            logger.info(f"  Total patients: {analysis['summary_statistics']['total_patients']}")
            logger.info(f"  Predicted positive: {analysis['summary_statistics']['predicted_positive']}")
            logger.info(f"  Mean probability: {analysis['summary_statistics']['mean_probability']:.3f}")
            logger.info(f"  Mean confidence: {analysis['summary_statistics']['mean_confidence']:.3f}")
            
            # Print results table
            print("\n" + "="*80)
            print("BATCH PREDICTION RESULTS")
            print("="*80)
            print(results_df[['patient_id', 'probability', 'prediction', 'risk_category', 'confidence']].to_string(index=False))
            
            # Save analysis
            if output_dir:
                with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
                    json.dump(analysis, f, indent=2)
        
    else:
        # Single patient prediction mode
        patient_id = os.path.basename(args.mri_dir) if args.patient_id is None else args.patient_id
        
        logger.info(f"Running single prediction for patient: {patient_id}")
        logger.info(f"  MRI directory: {args.mri_dir}")
        logger.info(f"  cfDNA features: {args.cfdna_features}")
        
        try:
            # Load patient data
            mri_data, cfdna_features = predictor.load_patient_data(args.mri_dir, args.cfdna_features)
            
            # Make prediction
            result = predictor.predict_single(mri_data, cfdna_features, return_features=args.return_features)
            
            # Display results
            print("\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            print(f"Patient ID: {patient_id}")
            print(f"Cancer Probability: {result['probability']:.3f}")
            print(f"Prediction: {'POSITIVE' if result['prediction'] else 'NEGATIVE'}")
            print(f"Risk Category: {result['risk_category']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Threshold Used: {result['threshold_used']}")
            print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
            
            if args.verbose:
                print(f"\nDetailed Information:")
                print(f"  MRI Shape: {result['data_shapes']['mri']}")
                print(f"  cfDNA Shape: {result['data_shapes']['cfdna']}")
                print(f"  Model Info: Epoch {model_info['epoch']}, Best AUC {model_info['best_metric']}")
                if 'raw_output' in result:
                    print(f"  Raw Output (logits): {result['raw_output']:.6f}")
            
            # Save results
            if args.output_file or output_dir:
                result_data = {
                    'patient_id': patient_id,
                    'prediction_timestamp': datetime.now().isoformat(),
                    'model_info': model_info,
                    'prediction_result': result,
                    'file_paths': {
                        'mri_directory': args.mri_dir,
                        'cfdna_features': args.cfdna_features,
                        'model_checkpoint': args.model_path
                    }
                }
                
                if args.output_file:
                    # Save to specified file
                    if args.output_file.endswith('.json'):
                        with open(args.output_file, 'w') as f:
                            json.dump(result_data, f, indent=2)
                    else:
                        with open(args.output_file, 'w') as f:
                            f.write(f"Patient ID: {patient_id}\n")
                            f.write(f"Cancer Probability: {result['probability']:.3f}\n")
                            f.write(f"Prediction: {'POSITIVE' if result['prediction'] else 'NEGATIVE'}\n")
                            f.write(f"Risk Category: {result['risk_category']}\n")
                            f.write(f"Confidence: {result['confidence']:.3f}\n")
                    
                    logger.info(f"Results saved to {args.output_file}")
                
                if output_dir:
                    # Save to output directory
                    with open(os.path.join(output_dir, 'prediction_result.json'), 'w') as f:
                        json.dump(result_data, f, indent=2)
                    
                    # Save simple CSV format
                    simple_result = pd.DataFrame([{
                        'patient_id': patient_id,
                        'probability': result['probability'],
                        'prediction': result['prediction'],
                        'risk_category': result['risk_category'],
                        'confidence': result['confidence']
                    }])
                    simple_result.to_csv(os.path.join(output_dir, 'prediction_summary.csv'), index=False)
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            print(f"\nError: {str(e)}")
            return
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\nPrediction completed in {total_time:.2f} seconds")
    
    if output_dir:
        logger.info(f"All results saved to: {output_dir}")
        print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced BEAM model prediction with comprehensive analysis')
    
    # Prediction mode
    parser.add_argument('--batch_mode', action='store_true',
                        help='Enable batch prediction mode')
    
    # Single patient arguments
    parser.add_argument('--mri_dir', type=str,
                        help='Directory containing T1.npy, T2.npy, DWI.npy (single mode)')
    parser.add_argument('--cfdna_features', type=str,
                        help='Path to cfDNA features numpy file (single mode)')
    parser.add_argument('--patient_id', type=str, default=None,
                        help='Patient identifier (optional)')
    
    # Batch prediction arguments
    parser.add_argument('--data_dir', type=str,
                        help='Root directory containing patient folders (batch mode)')
    parser.add_argument('--patient_list', type=str, default=None,
                        help='Text file with list of patient IDs to process (batch mode)')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--cfdna_dim', type=int, default=512,
                        help='cfDNA feature dimension')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    
    # Prediction settings
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--return_features', action='store_true',
                        help='Return intermediate features (single mode)')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                        help='Save prediction results to specific file (single mode)')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validation
    if args.batch_mode:
        if not args.data_dir:
            parser.error("--data_dir is required for batch mode")
        if not os.path.exists(args.data_dir):
            parser.error(f"Data directory not found: {args.data_dir}")
    else:
        if not args.mri_dir or not args.cfdna_features:
            parser.error("--mri_dir and --cfdna_features are required for single prediction mode")
        if not os.path.exists(args.mri_dir):
            parser.error(f"MRI directory not found: {args.mri_dir}")
        if not os.path.exists(args.cfdna_features):
            parser.error(f"cfDNA features file not found: {args.cfdna_features}")
    
    if not os.path.exists(args.model_path):
        parser.error(f"Model checkpoint not found: {args.model_path}")
    
    # Run prediction
    print("Starting BEAM prediction...")
    if args.batch_mode:
        print(f"Mode: Batch prediction")
        print(f"Data directory: {args.data_dir}")
    else:
        print(f"Mode: Single patient prediction")
        print(f"Patient: {args.patient_id or os.path.basename(args.mri_dir)}")
    
    main(args)