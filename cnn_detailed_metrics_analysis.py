'''
Detailed CNN 10-Fold Cross-Validation Analysis
Comprehensive per-class metrics and confusion matrix analysis
'''

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import os

# Create results directory
RESULTS_DIR = "CNN Detailed Analysis Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}/")

def calculate_per_class_metrics(y_true, y_pred):
    """Calculate detailed per-class metrics"""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class is present
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:  # Only benign
                tn = len(y_true)
                fp = fn = tp = 0
            else:  # Only malware
                tp = len(y_true)
                tn = fp = fn = 0
        else:
            tn = fp = fn = tp = 0
    
    # Per-class metrics
    metrics = {}
    
    # Benign (Class 0) metrics
    benign_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    benign_f1 = 2 * (benign_precision * benign_recall) / (benign_precision + benign_recall) if (benign_precision + benign_recall) > 0 else 0
    
    # Malware (Class 1) metrics  
    malware_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    malware_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    malware_f1 = 2 * (malware_precision * malware_recall) / (malware_precision + malware_recall) if (malware_precision + malware_recall) > 0 else 0
    
    # Overall metrics
    overall_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    overall_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'benign': {
            'precision': benign_precision,
            'recall': benign_recall,
            'f1': benign_f1
        },
        'malware': {
            'precision': malware_precision,
            'recall': malware_recall,
            'f1': malware_f1
        },
        'overall': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        }
    }
    
    return metrics

def analyze_cnn_10fold_detailed():
    """Comprehensive analysis of CNN 10-fold cross-validation results"""
    print("=== CNN 10-Fold Detailed Analysis ===")
    
    # Try to load the 10-fold results
    results_files = [
        '10-Fold CV Results/cv_results.pkl',
        'Archived/10-Fold CV Results/cv_results.pkl',
        'cnn_10fold_results.pkl'
    ]
    
    results_data = None
    for file_path in results_files:
        try:
            with open(file_path, 'rb') as f:
                results_data = pickle.load(f)
            print(f"✅ Loaded results from: {file_path}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            continue
    
    if results_data is None:
        print("❌ Could not find CNN 10-fold results")
        return
    
    print(f"Results structure: {list(results_data.keys())}")
    
    # Extract predictions and labels
    if 'all_predictions' in results_data and 'all_labels' in results_data:
        all_predictions = results_data['all_predictions']
        all_labels = results_data['all_labels']
        
        print(f"Total predictions: {len(all_predictions)}")
        print(f"Total labels: {len(all_labels)}")
        print(f"Class distribution - Benign: {list(all_labels).count(0)}, Malware: {list(all_labels).count(1)}")
        
        # Calculate comprehensive metrics
        detailed_metrics = calculate_per_class_metrics(all_labels, all_predictions)
        
        # Print detailed results
        print(f"\n=== COMPREHENSIVE CNN 10-FOLD RESULTS ===")
        print(f"Dataset: {len(all_labels)} samples total")
        print(f"Benign: {list(all_labels).count(0)} samples ({list(all_labels).count(0)/len(all_labels)*100:.1f}%)")
        print(f"Malware: {list(all_labels).count(1)} samples ({list(all_labels).count(1)/len(all_labels)*100:.1f}%)")
        
        # Confusion Matrix
        cm = detailed_metrics['confusion_matrix']
        tn, fp, fn, tp = detailed_metrics['tn'], detailed_metrics['fp'], detailed_metrics['fn'], detailed_metrics['tp']
        
        print(f"\n=== CONFUSION MATRIX ===")
        print(f"                Predicted")
        print(f"Actual    Benign  Malware")
        print(f"Benign      {tn:4d}     {fp:3d}")
        print(f"Malware     {fn:4d}     {tp:3d}")
        
        print(f"\nConfusion Matrix Breakdown:")
        print(f"  True Negatives (TN):  {tn:4d} - Correctly identified benign")
        print(f"  False Positives (FP): {fp:4d} - Benign incorrectly labeled as malware")
        print(f"  False Negatives (FN): {fn:4d} - Malware incorrectly labeled as benign")
        print(f"  True Positives (TP):  {tp:4d} - Correctly identified malware")
        
        # Per-class metrics
        print(f"\n=== PER-CLASS METRICS ===")
        
        print(f"\nBENIGN CLASS (Class 0):")
        print(f"  Precision: {detailed_metrics['benign']['precision']:.4f} ({detailed_metrics['benign']['precision']*100:.2f}%)")
        print(f"  Recall:    {detailed_metrics['benign']['recall']:.4f} ({detailed_metrics['benign']['recall']*100:.2f}%)")
        print(f"  F1-Score:  {detailed_metrics['benign']['f1']:.4f} ({detailed_metrics['benign']['f1']*100:.2f}%)")
        
        print(f"\nMALWARE CLASS (Class 1):")
        print(f"  Precision: {detailed_metrics['malware']['precision']:.4f} ({detailed_metrics['malware']['precision']*100:.2f}%)")
        print(f"  Recall:    {detailed_metrics['malware']['recall']:.4f} ({detailed_metrics['malware']['recall']*100:.2f}%)")
        print(f"  F1-Score:  {detailed_metrics['malware']['f1']:.4f} ({detailed_metrics['malware']['f1']*100:.2f}%)")
        
        # Overall metrics
        print(f"\n=== OVERALL METRICS ===")
        print(f"  Accuracy:  {detailed_metrics['overall']['accuracy']:.4f} ({detailed_metrics['overall']['accuracy']*100:.2f}%)")
        print(f"  Precision: {detailed_metrics['overall']['precision']:.4f} ({detailed_metrics['overall']['precision']*100:.2f}%)")
        print(f"  Recall:    {detailed_metrics['overall']['recall']:.4f} ({detailed_metrics['overall']['recall']*100:.2f}%)")
        print(f"  F1-Score:  {detailed_metrics['overall']['f1']:.4f} ({detailed_metrics['overall']['f1']*100:.2f}%)")
        
        # Additional analysis using sklearn
        print(f"\n=== SKLEARN CLASSIFICATION REPORT ===")
        report = classification_report(all_labels, all_predictions, 
                                     target_names=['Benign', 'Malware'], 
                                     digits=4, zero_division=0)
        print(report)
        
        # Class distribution analysis
        print(f"\n=== PERFORMANCE ANALYSIS ===")
        
        # Check for class imbalance bias
        baseline_acc = list(all_labels).count(0) / len(all_labels)
        print(f"Baseline accuracy (predict all benign): {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
        print(f"CNN accuracy: {detailed_metrics['overall']['accuracy']:.4f} ({detailed_metrics['overall']['accuracy']*100:.1f}%)")
        print(f"Improvement over baseline: {(detailed_metrics['overall']['accuracy'] - baseline_acc)*100:.1f} percentage points")
        
        # Malware detection effectiveness
        malware_recall = detailed_metrics['malware']['recall']
        if malware_recall > 0.95:
            effectiveness = "🏆 EXCELLENT"
        elif malware_recall > 0.90:
            effectiveness = "✅ VERY GOOD"
        elif malware_recall > 0.80:
            effectiveness = "⚠️ GOOD"
        elif malware_recall > 0.50:
            effectiveness = "⚠️ MODERATE"
        else:
            effectiveness = "❌ POOR"
        
        print(f"Malware detection effectiveness: {effectiveness} ({malware_recall*100:.1f}% recall)")
        
        # Create visualizations
        create_detailed_visualizations(detailed_metrics, all_labels, all_predictions)
        
        # Save detailed results
        save_detailed_results(detailed_metrics, all_labels, all_predictions)
        
        return detailed_metrics
    
    else:
        print(f"❌ Required keys not found in results. Available keys: {list(results_data.keys())}")
        return None

def create_detailed_visualizations(metrics, all_labels, all_predictions):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix Heatmap
    ax1 = axes[0, 0]
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malware'],
                yticklabels=['Benign', 'Malware'], ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    # 2. Per-Class Metrics Bar Chart
    ax2 = axes[0, 1]
    classes = ['Benign', 'Malware']
    precision_scores = [metrics['benign']['precision'], metrics['malware']['precision']]
    recall_scores = [metrics['benign']['recall'], metrics['malware']['recall']]
    f1_scores = [metrics['benign']['f1'], metrics['malware']['f1']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax2.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
    ax2.bar(x, recall_scores, width, label='Recall', alpha=0.8)
    ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
        ax2.text(i - width, p + 0.02, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Overall Metrics Radar Chart (simplified as bar chart)
    ax3 = axes[1, 0]
    overall_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    overall_values = [
        metrics['overall']['accuracy'],
        metrics['overall']['precision'],
        metrics['overall']['recall'],
        metrics['overall']['f1']
    ]
    
    bars = ax3.bar(overall_metrics, overall_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax3.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, value in zip(bars, overall_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Class Distribution and Performance Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
CNN 10-Fold Cross-Validation Summary

Dataset:
• Total samples: {len(all_labels)}
• Benign: {list(all_labels).count(0)} ({list(all_labels).count(0)/len(all_labels)*100:.1f}%)
• Malware: {list(all_labels).count(1)} ({list(all_labels).count(1)/len(all_labels)*100:.1f}%)

Confusion Matrix:
• True Negatives:  {metrics['tn']:4d}
• False Positives: {metrics['fp']:4d}
• False Negatives: {metrics['fn']:4d}
• True Positives:  {metrics['tp']:4d}

Key Results:
• Overall Accuracy: {metrics['overall']['accuracy']*100:.2f}%
• Malware Recall: {metrics['malware']['recall']*100:.2f}%
• Malware Precision: {metrics['malware']['precision']*100:.2f}%

Performance Assessment:
• Baseline (all benign): {list(all_labels).count(0)/len(all_labels)*100:.1f}%
• CNN improvement: +{(metrics['overall']['accuracy'] - list(all_labels).count(0)/len(all_labels))*100:.1f}pp
• Malware detection: {'🏆 EXCELLENT' if metrics['malware']['recall'] > 0.95 else '✅ VERY GOOD' if metrics['malware']['recall'] > 0.90 else '⚠️ GOOD'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'cnn_detailed_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Detailed analysis plot saved to: {plot_path}")

def save_detailed_results(metrics, all_labels, all_predictions):
    """Save detailed results to files"""
    
    # Save metrics to pickle
    results_path = os.path.join(RESULTS_DIR, 'cnn_detailed_metrics.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'all_labels': all_labels,
            'all_predictions': all_predictions
        }, f)
    
    # Save summary to text file
    summary_path = os.path.join(RESULTS_DIR, 'cnn_detailed_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CNN 10-Fold Cross-Validation Detailed Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"  Total samples: {len(all_labels)}\n")
        f.write(f"  Benign: {list(all_labels).count(0)} ({list(all_labels).count(0)/len(all_labels)*100:.1f}%)\n")
        f.write(f"  Malware: {list(all_labels).count(1)} ({list(all_labels).count(1)/len(all_labels)*100:.1f}%)\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Negatives (TN):  {metrics['tn']:4d}\n")
        f.write(f"  False Positives (FP): {metrics['fp']:4d}\n")
        f.write(f"  False Negatives (FN): {metrics['fn']:4d}\n")
        f.write(f"  True Positives (TP):  {metrics['tp']:4d}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        f.write(f"  Benign Class:\n")
        f.write(f"    Precision: {metrics['benign']['precision']:.4f}\n")
        f.write(f"    Recall:    {metrics['benign']['recall']:.4f}\n")
        f.write(f"    F1-Score:  {metrics['benign']['f1']:.4f}\n")
        f.write(f"  Malware Class:\n")
        f.write(f"    Precision: {metrics['malware']['precision']:.4f}\n")
        f.write(f"    Recall:    {metrics['malware']['recall']:.4f}\n")
        f.write(f"    F1-Score:  {metrics['malware']['f1']:.4f}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {metrics['overall']['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['overall']['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['overall']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['overall']['f1']:.4f}\n")
    
    print(f"✅ Detailed results saved to: {RESULTS_DIR}/")

if __name__ == "__main__":
    print("=== CNN 10-Fold Detailed Metrics Analysis ===")
    analyze_cnn_10fold_detailed()
    print(f"\n✅ Analysis complete. All results saved to: {RESULTS_DIR}/")
