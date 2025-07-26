import argparse
import os
import json
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, precision_recall_curve, average_precision_score,
    roc_curve, auc, calibration_curve, brier_score_loss
)
from scipy import stats
from datetime import datetime

from models import BEAM
from data import MultiModalDataset
from utils import (
    load_checkpoint, calculate_metrics, plot_roc_curve, 
    plot_confusion_matrix, get_logger, create_dirs
)

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model, device, threshold=0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        
    def evaluate_model(self, dataloader, return_patient_ids=False):
        """Evaluate model and return detailed results"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Evaluating"):
                if len(batch_data) == 4:  # If patient IDs are included
                    mri_data, cfdna_data, labels, patient_ids = batch_data
                    all_patient_ids.extend(patient_ids)
                else:
                    mri_data, cfdna_data, labels = batch_data
                
                mri_data = mri_data.to(self.device, non_blocking=True)
                cfdna_data = cfdna_data.to(self.device, non_blocking=True)
                
                outputs = self.model(mri_data, cfdna_data)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > self.threshold).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        results = {
            'labels': np.array(all_labels),
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs)
        }
        
        if return_patient_ids and all_patient_ids:
            results['patient_ids'] = all_patient_ids
            
        return results
    
    def threshold_analysis(self, labels, probs, save_path=None):
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics_by_threshold = {
            'threshold': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'f1': [],
            'youden_j': []
        }
        
        for thresh in thresholds:
            preds = (probs > thresh).astype(int)
            metrics = calculate_metrics(labels, preds, probs)
            
            metrics_by_threshold['threshold'].append(thresh)
            metrics_by_threshold['accuracy'].append(metrics['accuracy'])
            metrics_by_threshold['precision'].append(metrics['precision'])
            metrics_by_threshold['recall'].append(metrics['recall'])
            metrics_by_threshold['specificity'].append(metrics['specificity'])
            metrics_by_threshold['f1'].append(metrics['f1'])
            
            # Youden's J statistic
            youden_j = metrics['recall'] + metrics['specificity'] - 1
            metrics_by_threshold['youden_j'].append(youden_j)
        
        # Find optimal threshold
        optimal_idx = np.argmax(metrics_by_threshold['youden_j'])
        optimal_threshold = thresholds[optimal_idx]
        
        # Plot threshold analysis
        if save_path:
            self._plot_threshold_analysis(metrics_by_threshold, optimal_threshold, save_path)
        
        return pd.DataFrame(metrics_by_threshold), optimal_threshold
    
    def _plot_threshold_analysis(self, metrics, optimal_threshold, save_path):
        """Plot threshold analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy, Precision, Recall
        axes[0, 0].plot(metrics['threshold'], metrics['accuracy'], 'b-', label='Accuracy', linewidth=2)
        axes[0, 0].plot(metrics['threshold'], metrics['precision'], 'r-', label='Precision', linewidth=2)
        axes[0, 0].plot(metrics['threshold'], metrics['recall'], 'g-', label='Recall', linewidth=2)
        axes[0, 0].axvline(optimal_threshold, color='orange', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[0, 1].plot(metrics['threshold'], metrics['f1'], 'purple', linewidth=2)
        axes[0, 1].axvline(optimal_threshold, color='orange', linestyle='--')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sensitivity vs Specificity
        axes[1, 0].plot(metrics['threshold'], metrics['recall'], 'g-', label='Sensitivity', linewidth=2)
        axes[1, 0].plot(metrics['threshold'], metrics['specificity'], 'm-', label='Specificity', linewidth=2)
        axes[1, 0].axvline(optimal_threshold, color='orange', linestyle='--')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Sensitivity vs Specificity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Youden's J
        axes[1, 1].plot(metrics['threshold'], metrics['youden_j'], 'orange', linewidth=2)
        axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel("Youden's J")
        axes[1, 1].set_title("Youden's J Statistic")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def stratified_analysis(self, labels, probs, stratify_by=None, save_path=None):
        """Perform stratified analysis by subgroups"""
        if stratify_by is None:
            return None
        
        results = {}
        unique_groups = np.unique(stratify_by)
        
        for group in unique_groups:
            mask = stratify_by == group
            group_labels = labels[mask]
            group_probs = probs[mask]
            group_preds = (group_probs > self.threshold).astype(int)
            
            if len(np.unique(group_labels)) > 1:  # Ensure both classes present
                metrics = calculate_metrics(group_labels, group_preds, group_probs)
                results[group] = {
                    'n_samples': len(group_labels),
                    'n_positive': np.sum(group_labels),
                    'metrics': metrics
                }
        
        return results
    
    def calibration_analysis(self, labels, probs, save_path=None):
        """Analyze model calibration"""
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probs, n_bins=10
        )
        
        # Brier score
        brier_score = brier_score_loss(labels, probs)
        
        # Plot calibration curve
        if save_path:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                   linewidth=2, label=f'Model (Brier Score: {brier_score:.3f})')
            ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Calibration Plot")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'brier_score': brier_score,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    
    def precision_recall_analysis(self, labels, probs, save_path=None):
        """Analyze precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        ap_score = average_precision_score(labels, probs)
        
        if save_path:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, linewidth=2, 
                   label=f'PR curve (AP: {ap_score:.3f})')
            ax.axhline(y=np.mean(labels), color='gray', linestyle='--', 
                      label=f'Baseline: {np.mean(labels):.3f}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'average_precision': ap_score,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    
    def confidence_interval_analysis(self, labels, probs, n_bootstrap=1000, confidence=0.95):
        """Bootstrap confidence intervals for metrics"""
        from sklearn.utils import resample
        
        metrics_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(len(labels)), random_state=None)
            boot_labels = labels[indices]
            boot_probs = probs[indices]
            boot_preds = (boot_probs > self.threshold).astype(int)
            
            # Calculate metrics if both classes present
            if len(np.unique(boot_labels)) > 1:
                metrics = calculate_metrics(boot_labels, boot_preds, boot_probs)
                metrics_bootstrap.append(metrics)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_results = {}
        if metrics_bootstrap:
            df = pd.DataFrame(metrics_bootstrap)
            for metric in df.columns:
                ci_results[metric] = {
                    'mean': df[metric].mean(),
                    'lower': np.percentile(df[metric], lower_percentile),
                    'upper': np.percentile(df[metric], upper_percentile)
                }
        
        return ci_results


def generate_comprehensive_report(evaluator, results, args, save_dir):
    """Generate comprehensive evaluation report"""
    labels = results['labels']
    probs = results['probabilities']
    preds = results['predictions']
    
    report = {
        'evaluation_summary': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'test_data_path': args.test_data,
            'threshold': args.threshold,
            'total_samples': len(labels),
            'positive_samples': int(np.sum(labels)),
            'negative_samples': int(len(labels) - np.sum(labels)),
            'positive_ratio': float(np.mean(labels))
        }
    }
    
    # Basic metrics
    basic_metrics = calculate_metrics(labels, preds, probs)
    report['basic_metrics'] = basic_metrics
    
    # Threshold analysis
    if args.threshold_analysis:
        threshold_df, optimal_threshold = evaluator.threshold_analysis(
            labels, probs, os.path.join(save_dir, 'threshold_analysis.png')
        )
        threshold_df.to_csv(os.path.join(save_dir, 'threshold_analysis.csv'), index=False)
        report['threshold_analysis'] = {
            'optimal_threshold': float(optimal_threshold),
            'current_threshold': args.threshold
        }
    
    # Calibration analysis
    if args.calibration_analysis:
        calibration_results = evaluator.calibration_analysis(
            labels, probs, os.path.join(save_dir, 'calibration_plot.png')
        )
        report['calibration'] = {
            'brier_score': float(calibration_results['brier_score'])
        }
    
    # Precision-Recall analysis
    if args.pr_analysis:
        pr_results = evaluator.precision_recall_analysis(
            labels, probs, os.path.join(save_dir, 'precision_recall_curve.png')
        )
        report['precision_recall'] = {
            'average_precision': float(pr_results['average_precision'])
        }
    
    # Confidence intervals
    if args.confidence_intervals:
        ci_results = evaluator.confidence_interval_analysis(labels, probs)
        report['confidence_intervals'] = ci_results
    
    # Classification report
    class_report = classification_report(labels, preds, output_dict=True)
    report['classification_report'] = class_report
    
    return report


def plot_error_analysis(labels, probs, preds, save_path):
    """Plot error analysis"""
    # False positives and false negatives
    fp_mask = (labels == 0) & (preds == 1)
    fn_mask = (labels == 1) & (preds == 0)
    tp_mask = (labels == 1) & (preds == 1)
    tn_mask = (labels == 0) & (preds == 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Probability distribution by prediction type
    axes[0, 0].hist(probs[tp_mask], bins=20, alpha=0.7, label='True Positives', color='green')
    axes[0, 0].hist(probs[fp_mask], bins=20, alpha=0.7, label='False Positives', color='red')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Probability Distribution - Positive Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(probs[tn_mask], bins=20, alpha=0.7, label='True Negatives', color='blue')
    axes[0, 1].hist(probs[fn_mask], bins=20, alpha=0.7, label='False Negatives', color='orange')
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Probability Distribution - Negative Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error rates by probability bins
    prob_bins = np.linspace(0, 1, 11)
    bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
    fp_rates = []
    fn_rates = []
    
    for i in range(len(prob_bins) - 1):
        mask = (probs >= prob_bins[i]) & (probs < prob_bins[i + 1])
        if np.sum(mask) > 0:
            bin_labels = labels[mask]
            bin_preds = preds[mask]
            
            if len(np.unique(bin_labels)) > 1:
                fp_rate = np.sum((bin_labels == 0) & (bin_preds == 1)) / np.sum(bin_labels == 0) if np.sum(bin_labels == 0) > 0 else 0
                fn_rate = np.sum((bin_labels == 1) & (bin_preds == 0)) / np.sum(bin_labels == 1) if np.sum(bin_labels == 1) > 0 else 0
            else:
                fp_rate = fn_rate = 0
        else:
            fp_rate = fn_rate = 0
            
        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)
    
    axes[1, 0].bar(bin_centers, fp_rates, width=0.08, alpha=0.7, color='red', label='False Positive Rate')
    axes[1, 0].set_xlabel('Probability Bin')
    axes[1, 0].set_ylabel('Error Rate')
    axes[1, 0].set_title('False Positive Rate by Probability Bin')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(bin_centers, fn_rates, width=0.08, alpha=0.7, color='orange', label='False Negative Rate')
    axes[1, 1].set_xlabel('Probability Bin')
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].set_title('False Negative Rate by Probability Bin')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    start_time = time.time()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    create_dirs([eval_dir, os.path.join(eval_dir, 'plots')])
    
    # Logger
    logger = get_logger('BEAM-Eval', os.path.join(eval_dir, 'evaluate.log'))
    logger.info(f"Evaluation directory: {eval_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Save configuration
    with open(os.path.join(eval_dir, 'eval_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Load dataset
    logger.info("Loading test dataset...")
    dataset = MultiModalDataset(args.test_data, args.labels_file)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.info(f"Test dataset size: {len(dataset)}")
    labels = np.array(dataset.labels)
    logger.info(f"Positive samples: {np.sum(labels)} ({np.mean(labels):.3f})")
    
    # Load model
    logger.info("Loading model...")
    model = BEAM(
        cfdna_dim=args.cfdna_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0  # No dropout during evaluation
    ).to(device)
    
    checkpoint = load_checkpoint(args.model_path, model, device=device)
    logger.info(f"Model loaded from epoch {checkpoint['epoch']} with best AUC: {checkpoint['best_metric']:.4f}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device, args.threshold)
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_start_time = time.time()
    results = evaluator.evaluate_model(dataloader)
    eval_time = time.time() - eval_start_time
    
    labels = results['labels']
    preds = results['predictions']
    probs = results['probabilities']
    
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Basic metrics
    basic_metrics = calculate_metrics(labels, preds, probs)
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Dataset Size: {len(labels)}")
    logger.info(f"Positive Ratio: {np.mean(labels):.3f}")
    logger.info(f"Threshold Used: {args.threshold}")
    logger.info("\nBasic Metrics:")
    for metric, value in basic_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Generate comprehensive report
    logger.info("\nGenerating comprehensive analysis...")
    report = generate_comprehensive_report(evaluator, results, args, os.path.join(eval_dir, 'plots'))
    
    # Save detailed results
    with open(os.path.join(eval_dir, 'comprehensive_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save basic metrics
    pd.DataFrame([basic_metrics]).to_csv(
        os.path.join(eval_dir, 'basic_metrics.csv'), index=False
    )
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': preds,
        'cancer_probability': probs
    })
    
    if 'patient_ids' in results:
        predictions_df['patient_id'] = results['patient_ids']
    
    predictions_df.to_csv(os.path.join(eval_dir, 'predictions.csv'), index=False)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # ROC curve
    plot_roc_curve(labels, probs, os.path.join(eval_dir, 'plots', 'roc_curve.png'))
    
    # Confusion matrix
    plot_confusion_matrix(labels, preds, os.path.join(eval_dir, 'plots', 'confusion_matrix.png'))
    
    # Error analysis
    plot_error_analysis(labels, probs, preds, os.path.join(eval_dir, 'plots', 'error_analysis.png'))
    
    # Additional analyses based on flags
    if args.threshold_analysis:
        logger.info("Optimal threshold: {:.3f}".format(report.get('threshold_analysis', {}).get('optimal_threshold', 'N/A')))
    
    if args.calibration_analysis:
        logger.info("Brier score: {:.4f}".format(report.get('calibration', {}).get('brier_score', 'N/A')))
    
    if args.pr_analysis:
        logger.info("Average precision: {:.4f}".format(report.get('precision_recall', {}).get('average_precision', 'N/A')))
    
    if args.confidence_intervals:
        ci_auc = report.get('confidence_intervals', {}).get('auc', {})
        if ci_auc:
            logger.info(f"AUC 95% CI: [{ci_auc.get('lower', 'N/A'):.4f}, {ci_auc.get('upper', 'N/A'):.4f}]")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\nTotal evaluation time: {total_time:.2f} seconds")
    logger.info(f"Results saved to: {eval_dir}")
    
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive BEAM model evaluation with advanced analytics')
    
    # Data arguments
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to labels CSV file')
    
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
    
    # Evaluation settings
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    
    # Advanced analysis options
    parser.add_argument('--threshold_analysis', action='store_true',
                        help='Perform threshold optimization analysis')
    parser.add_argument('--calibration_analysis', action='store_true',
                        help='Perform model calibration analysis')
    parser.add_argument('--pr_analysis', action='store_true',
                        help='Perform precision-recall analysis')
    parser.add_argument('--confidence_intervals', action='store_true',
                        help='Calculate bootstrap confidence intervals')
    parser.add_argument('--stratified_analysis', action='store_true',
                        help='Perform stratified analysis by subgroups')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save evaluation plots')
    parser.add_argument('--detailed_report', action='store_true', default=True,
                        help='Generate detailed evaluation report')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    print("Starting comprehensive evaluation...")
    if args.threshold_analysis:
        print("  - Threshold optimization enabled")
    if args.calibration_analysis:
        print("  - Calibration analysis enabled") 
    if args.pr_analysis:
        print("  - Precision-recall analysis enabled")
    if args.confidence_intervals:
        print("  - Bootstrap confidence intervals enabled")
    
    report = main(args)
    print(f"\\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")