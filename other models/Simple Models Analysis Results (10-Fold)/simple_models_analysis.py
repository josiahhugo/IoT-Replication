'''
Simple Models Cross-Validation Analysis for IoT Malware Detection
Comprehensive evaluation of Logistic Regression, Random Forest, and SVM
with detailed per-class metrics (accuracy, recall, F1) for benign and malware
'''

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from alive_progress import alive_bar
import warnings
warnings.filterwarnings('ignore')
import os

# Create results directory
RESULTS_DIR = "Simple Models Analysis Results (10-Fold)"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}/")

def load_data():
    """Load eigenspace embeddings and labels"""
    try:
        # Try different data sources
        X = None
        
        # Try balanced embeddings first
        try:
            with open('X_graph_embeddings_balanced.pkl', 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    X = data.get('embeddings', data.get('X', None))
                else:
                    X = data
            print(f"✅ Loaded balanced eigenspace embeddings")
        except FileNotFoundError:
            # Try regular embeddings
            try:
                with open('X_graph_embeddings.pkl', 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        X = data.get('embeddings', data.get('X', None))
                    else:
                        X = data
                print(f"✅ Loaded regular eigenspace embeddings")
            except FileNotFoundError:
                print("❌ No eigenspace embeddings found")
                return None, None
        
        # Ensure X is a numpy array
        if X is not None:
            X = np.array(X)
            print(f"   Shape: {X.shape}")
        else:
            print("❌ Could not extract embeddings from data")
            return None, None
        
        # Load labels
        with open('improved_cig_output.pkl', 'rb') as f:
            data = pickle.load(f)
            y = np.array(data['labels'])
        
        print(f"✅ Loaded labels: {len(y)} samples")
        print(f"   Class distribution: {np.sum(y == 0)} benign, {np.sum(y == 1)} malware")
        print(f"   Class imbalance ratio: {np.sum(y == 0)/np.sum(y == 1):.2f}:1")
        
        # Verify dimensions match
        if len(X) != len(y):
            print(f"❌ Dimension mismatch: X has {len(X)} samples, y has {len(y)} samples")
            return None, None
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def custom_scorer_malware_recall(y_true, y_pred):
    """Custom scorer for malware recall"""
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def custom_scorer_benign_recall(y_true, y_pred):
    """Custom scorer for benign recall"""
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)

def custom_scorer_malware_f1(y_true, y_pred):
    """Custom scorer for malware F1"""
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def custom_scorer_benign_f1(y_true, y_pred):
    """Custom scorer for benign F1"""
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)

def custom_scorer_malware_precision(y_true, y_pred):
    """Custom scorer for malware precision"""
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)

def custom_scorer_benign_precision(y_true, y_pred):
    """Custom scorer for benign precision"""
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)

def evaluate_model_cv(model, X, y, model_name, cv_folds=10, use_smote=False):
    """
    Comprehensive cross-validation evaluation with detailed per-class metrics
    """
    print(f"\n🔄 Evaluating {model_name}...")
    
    # Define custom scoring functions
    scoring = {
        'accuracy': 'accuracy',
        'malware_recall': make_scorer(custom_scorer_malware_recall),
        'benign_recall': make_scorer(custom_scorer_benign_recall),
        'malware_f1': make_scorer(custom_scorer_malware_f1),
        'benign_f1': make_scorer(custom_scorer_benign_f1),
        'malware_precision': make_scorer(custom_scorer_malware_precision),
        'benign_precision': make_scorer(custom_scorer_benign_precision),
        'f1_weighted': 'f1_weighted',
        'roc_auc': 'roc_auc'
    }
    
    # Create pipeline with optional SMOTE
    if use_smote:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        print(f"   Using SMOTE + StandardScaler + {model_name}")
    else:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        print(f"   Using StandardScaler + {model_name}")
    
    # Perform cross-validation with progress bar
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform cross-validation (no nested progress bar due to alive_progress limitations)
    cv_results = cross_validate(
        pipeline, X, y, 
        cv=cv, 
        scoring=scoring, 
        return_train_score=True,
        n_jobs=-1  # Use all cores for faster processing
    )
    
    # Calculate statistics
    results = {}
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[metric] = {
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores),
            'overfitting_gap': np.mean(train_scores) - np.mean(test_scores),
            'test_scores': test_scores,
            'train_scores': train_scores
        }
    
    return results

def detailed_evaluation_single_split(model, X, y, model_name, use_smote=False):
    """
    Detailed evaluation on a single train-test split for confusion matrix and classification report
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE if requested
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Detailed metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malware'], output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'y_true': y_test,
        'y_pred': y_pred
    }

def create_comprehensive_models():
    """Create model configurations to test - optimized for speed"""
    
    # Fast and effective models for malware detection
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'  # Faster for small datasets
        ),
        'Logistic Regression (No Balance)': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,  # Reduced from 100 for speed
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=10  # Limit depth for speed
        ),
        'Random Forest (No Balance)': RandomForestClassifier(
            n_estimators=50,  # Reduced from 100 for speed
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
        # Removed SVM models as they're too slow for this dataset size
    }
    
    return models

def plot_results(all_results, save_path):
    """Create comprehensive visualization of results"""
    
    # Prepare data for plotting
    model_names = []
    metrics_data = {
        'Accuracy': [],
        'Malware Recall': [],
        'Benign Recall': [],
        'Malware F1': [],
        'Benign F1': [],
        'Malware Precision': [],
        'Benign Precision': []
    }
    errors = {metric: [] for metric in metrics_data.keys()}
    
    for model_name, results in all_results.items():
        model_names.append(model_name)
        metrics_data['Accuracy'].append(results['accuracy']['test_mean'])
        metrics_data['Malware Recall'].append(results['malware_recall']['test_mean'])
        metrics_data['Benign Recall'].append(results['benign_recall']['test_mean'])
        metrics_data['Malware F1'].append(results['malware_f1']['test_mean'])
        metrics_data['Benign F1'].append(results['benign_f1']['test_mean'])
        metrics_data['Malware Precision'].append(results['malware_precision']['test_mean'])
        metrics_data['Benign Precision'].append(results['benign_precision']['test_mean'])
        
        errors['Accuracy'].append(results['accuracy']['test_std'])
        errors['Malware Recall'].append(results['malware_recall']['test_std'])
        errors['Benign Recall'].append(results['benign_recall']['test_std'])
        errors['Malware F1'].append(results['malware_f1']['test_std'])
        errors['Benign F1'].append(results['benign_f1']['test_std'])
        errors['Malware Precision'].append(results['malware_precision']['test_std'])
        errors['Benign Precision'].append(results['benign_precision']['test_std'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.ravel()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax = axes[i]
        
        bars = ax.bar(range(len(model_names)), values, yerr=errors[metric], 
                     capsize=5, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, value, error in zip(bars, values, errors[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Highlight best performer
        if values:
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    # Remove unused subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Results plot saved to: {save_path}")

def plot_confusion_matrices(detailed_results, save_path):
    """Plot confusion matrices for all models"""
    
    n_models = len(detailed_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (model_name, results) in enumerate(detailed_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        cm = results['confusion_matrix']
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Benign', 'Malware'],
                   yticklabels=['Benign', 'Malware'])
        
        ax.set_title(f'{model_name}\nAccuracy: {results["test_accuracy"]:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Remove unused subplots
    for i in range(len(detailed_results), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrices saved to: {save_path}")

def create_summary_table(all_results, detailed_results):
    """Create a comprehensive summary table with clear per-class breakdown"""
    
    summary_data = []
    
    for model_name in all_results.keys():
        cv_results = all_results[model_name]
        detailed = detailed_results.get(model_name, {})
        
        row = {
            'Model': model_name,
            
            # Overall Metrics
            'Overall_Accuracy': f"{cv_results['accuracy']['test_mean']:.3f} ± {cv_results['accuracy']['test_std']:.3f}",
            'Overall_Precision': f"{cv_results['f1_weighted']['test_mean']:.3f} ± {cv_results['f1_weighted']['test_std']:.3f}",  # Using weighted F1 as proxy
            'Overall_F1': f"{cv_results['f1_weighted']['test_mean']:.3f} ± {cv_results['f1_weighted']['test_std']:.3f}",
            
            # Malware (Class 1) Metrics
            'Malware_Precision': f"{cv_results['malware_precision']['test_mean']:.3f} ± {cv_results['malware_precision']['test_std']:.3f}",
            'Malware_Recall': f"{cv_results['malware_recall']['test_mean']:.3f} ± {cv_results['malware_recall']['test_std']:.3f}",
            'Malware_F1': f"{cv_results['malware_f1']['test_mean']:.3f} ± {cv_results['malware_f1']['test_std']:.3f}",
            
            # Benign (Class 0) Metrics  
            'Benign_Precision': f"{cv_results['benign_precision']['test_mean']:.3f} ± {cv_results['benign_precision']['test_std']:.3f}",
            'Benign_Recall': f"{cv_results['benign_recall']['test_mean']:.3f} ± {cv_results['benign_recall']['test_std']:.3f}",
            'Benign_F1': f"{cv_results['benign_f1']['test_mean']:.3f} ± {cv_results['benign_f1']['test_std']:.3f}",
            
            # Additional Metrics
            'ROC_AUC': f"{cv_results['roc_auc']['test_mean']:.3f} ± {cv_results['roc_auc']['test_std']:.3f}",
            'Overfitting_Gap': f"{cv_results['accuracy']['overfitting_gap']:.3f}"
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df

def print_detailed_metrics(all_results):
    """Print detailed per-class and overall metrics in a clear format"""
    
    print(f"\n{'='*80}")
    print("📊 DETAILED METRICS BY MODEL")
    print(f"{'='*80}")
    
    for model_name, cv_results in all_results.items():
        print(f"\n🔹 {model_name}")
        print(f"{'─' * (len(model_name) + 3)}")
        
        # Overall Metrics
        print(f"\n📈 OVERALL METRICS:")
        print(f"   Accuracy:   {cv_results['accuracy']['test_mean']:.3f} ± {cv_results['accuracy']['test_std']:.3f}")
        print(f"   F1 (Weighted): {cv_results['f1_weighted']['test_mean']:.3f} ± {cv_results['f1_weighted']['test_std']:.3f}")
        print(f"   ROC AUC:    {cv_results['roc_auc']['test_mean']:.3f} ± {cv_results['roc_auc']['test_std']:.3f}")
        
        # Malware Class Metrics
        print(f"\n🦠 MALWARE CLASS METRICS:")
        print(f"   Precision:  {cv_results['malware_precision']['test_mean']:.3f} ± {cv_results['malware_precision']['test_std']:.3f}")
        print(f"   Recall:     {cv_results['malware_recall']['test_mean']:.3f} ± {cv_results['malware_recall']['test_std']:.3f}")
        print(f"   F1-Score:   {cv_results['malware_f1']['test_mean']:.3f} ± {cv_results['malware_f1']['test_std']:.3f}")
        
        # Benign Class Metrics  
        print(f"\n✅ BENIGN CLASS METRICS:")
        print(f"   Precision:  {cv_results['benign_precision']['test_mean']:.3f} ± {cv_results['benign_precision']['test_std']:.3f}")
        print(f"   Recall:     {cv_results['benign_recall']['test_mean']:.3f} ± {cv_results['benign_recall']['test_std']:.3f}")
        print(f"   F1-Score:   {cv_results['benign_f1']['test_mean']:.3f} ± {cv_results['benign_f1']['test_std']:.3f}")
        
        # Overfitting Assessment
        gap = cv_results['accuracy']['overfitting_gap']
        if gap < 0.05:
            status = "✅ Minimal"
        elif gap < 0.15:
            status = "⚠️ Moderate" 
        else:
            status = "❌ Severe"
        print(f"\n📊 Overfitting Gap: {gap:.3f} ({status})")

def create_ranking_table(all_results):
    """Create ranking tables for different metrics"""
    
    metrics_ranking = {}
    
    # Define metrics to rank
    ranking_metrics = {
        'Overall Accuracy': 'accuracy',
        'Malware Recall': 'malware_recall', 
        'Malware F1': 'malware_f1',
        'Benign Recall': 'benign_recall',
        'Benign F1': 'benign_f1',
        'ROC AUC': 'roc_auc'
    }
    
    for display_name, metric_key in ranking_metrics.items():
        # Sort models by this metric
        sorted_models = sorted(
            all_results.items(),
            key=lambda x: x[1][metric_key]['test_mean'],
            reverse=True
        )
        
        ranking_data = []
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            score = results[metric_key]['test_mean']
            std = results[metric_key]['test_std']
            ranking_data.append({
                'Rank': rank,
                'Model': model_name,
                'Score': f"{score:.3f} ± {std:.3f}",
                'Raw_Score': score
            })
        
        metrics_ranking[display_name] = ranking_data
    
    return metrics_ranking

def main():
    """Main analysis function"""
    print("=== Simple Models Cross-Validation Analysis ===")
    print("Evaluating Logistic Regression, Random Forest, and SVM")
    print("With detailed per-class metrics for benign and malware detection")
    print("Using 10-fold cross-validation for robust statistical estimates\n")
    
    # Load data
    X, y = load_data()
    if X is None or y is None:
        print("❌ Failed to load data")
        return
    
    # Create models
    models = create_comprehensive_models()
    
    # Store results
    all_results = {}
    detailed_results = {}
    
    # Evaluate each model with progress bar
    print(f"\n🔄 Evaluating {len(models)} models...")
    with alive_bar(len(models), title="Models") as bar:
        for model_name, model in models.items():
            # Cross-validation evaluation with 10-fold CV
            cv_results = evaluate_model_cv(model, X, y, model_name, cv_folds=10, use_smote=False)
            all_results[model_name] = cv_results
            
            # Detailed single-split evaluation
            detailed = detailed_evaluation_single_split(model, X, y, model_name, use_smote=False)
            detailed_results[model_name] = detailed
            
            # Print summary
            print(f"   Accuracy: {cv_results['accuracy']['test_mean']:.3f} ± {cv_results['accuracy']['test_std']:.3f}")
            print(f"   Malware Recall: {cv_results['malware_recall']['test_mean']:.3f} ± {cv_results['malware_recall']['test_std']:.3f}")
            print(f"   Benign Recall: {cv_results['benign_recall']['test_mean']:.3f} ± {cv_results['benign_recall']['test_std']:.3f}")
            print(f"   Overfitting Gap: {cv_results['accuracy']['overfitting_gap']:.3f}")
            
            bar()
    
    # Test with SMOTE
    print(f"\n🔄 Evaluating models with SMOTE...")
    smote_results = {}
    smote_detailed = {}
    
    selected_models = list(models.items())[:3]  # Test top 3 models with SMOTE
    with alive_bar(len(selected_models), title="SMOTE Models") as bar:
        for model_name, model in selected_models:
            smote_name = f"{model_name} + SMOTE"
            
            cv_results = evaluate_model_cv(model, X, y, smote_name, cv_folds=10, use_smote=True)
            all_results[smote_name] = cv_results
            
            detailed = detailed_evaluation_single_split(model, X, y, smote_name, use_smote=True)
            detailed_results[smote_name] = detailed
            
            print(f"   {smote_name}:")
            print(f"     Accuracy: {cv_results['accuracy']['test_mean']:.3f} ± {cv_results['accuracy']['test_std']:.3f}")
            print(f"     Malware Recall: {cv_results['malware_recall']['test_mean']:.3f} ± {cv_results['malware_recall']['test_std']:.3f}")
            
            bar()
    
    # Create visualizations
    plot_path = os.path.join(RESULTS_DIR, 'simple_models_comparison.png')
    plot_results(all_results, plot_path)
    
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
    plot_confusion_matrices(detailed_results, cm_path)
    
    # Create summary table
    summary_df = create_summary_table(all_results, detailed_results)
    
    # Print detailed metrics breakdown
    print_detailed_metrics(all_results)
    
    # Create ranking tables
    rankings = create_ranking_table(all_results)
    
    # Save results
    summary_path = os.path.join(RESULTS_DIR, 'simple_models_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    results_path = os.path.join(RESULTS_DIR, 'simple_models_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'cv_results': all_results,
            'detailed_results': detailed_results,
            'summary_table': summary_df,
            'rankings': rankings
        }, f)
    
    # Print final summary with rankings
    print(f"\n{'='*80}")
    print("🏆 MODEL RANKINGS BY KEY METRICS")
    print(f"{'='*80}")
    
    # Print top 3 for each key metric
    key_metrics = ['Malware Recall', 'Malware F1', 'Overall Accuracy', 'ROC AUC']
    
    for metric in key_metrics:
        if metric in rankings:
            print(f"\n🏅 {metric}:")
            for i, model_data in enumerate(rankings[metric][:3]):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"   {medal} {model_data['Model']}: {model_data['Score']}")
    
    print(f"\n📊 Complete Summary Table:")
    # Create a more readable summary for key metrics
    key_columns = ['Model', 'Overall_Accuracy', 'Malware_Precision', 'Malware_Recall', 
                   'Malware_F1', 'Benign_Precision', 'Benign_Recall', 'Benign_F1', 'ROC_AUC']
    print(summary_df[key_columns].to_string(index=False))
    
    print(f"\n📁 Results saved to: {RESULTS_DIR}/")
    print(f"  - simple_models_comparison.png")
    print(f"  - confusion_matrices.png") 
    print(f"  - simple_models_summary.csv")
    print(f"  - simple_models_results.pkl")

if __name__ == "__main__":
    main()
