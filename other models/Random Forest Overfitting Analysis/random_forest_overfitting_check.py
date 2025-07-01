'''
Random Forest Overfitting Diagnostic for IoT Malware Detection
Comprehensive analysis to detect if Random Forest is overfitting
'''

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import os

# Create results directory
RESULTS_DIR = "Random Forest Overfitting Analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}/")

def load_data():
    """Load eigenspace embeddings and labels"""
    try:
        # Try balanced embeddings first
        with open('X_graph_embeddings_balanced.pkl', 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                X = data.get('embeddings', data.get('X', None))
            else:
                X = data
        print(f"✅ Loaded balanced eigenspace embeddings")
        
        X = np.array(X)
        print(f"   Shape: {X.shape}")
        
        # Load labels
        with open('improved_cig_output.pkl', 'rb') as f:
            data = pickle.load(f)
            y = np.array(data['labels'])
        
        print(f"✅ Loaded labels: {len(y)} samples")
        print(f"   Class distribution: {np.sum(y == 0)} benign, {np.sum(y == 1)} malware")
        print(f"   Class imbalance ratio: {np.sum(y == 0)/np.sum(y == 1):.2f}:1")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def test_random_forest_overfitting(X, y):
    """Comprehensive overfitting analysis for Random Forest"""
    
    print("\n" + "="*80)
    print("🔍 RANDOM FOREST OVERFITTING DIAGNOSTIC")
    print("="*80)
    
    results = {}
    
    # Test 1: Different Random Forest configurations
    print("\n📊 Test 1: Different Random Forest Configurations")
    rf_configs = {
        'RF_Default': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'RF_Limited_Depth': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'RF_Few_Trees': RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced'),
        'RF_No_Balance': RandomForestClassifier(n_estimators=100, random_state=42),
        'RF_Min_Samples': RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=5, random_state=42, class_weight='balanced')
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for config_name, rf in rf_configs.items():
        print(f"\n🔹 {config_name}")
        
        fold_results = {
            'train_acc': [], 'test_acc': [],
            'train_f1': [], 'test_f1': [],
            'malware_recall': [], 'malware_precision': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_smote)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            rf.fit(X_train_scaled, y_train_smote)
            
            # Predictions
            y_train_pred = rf.predict(X_train_scaled)
            y_test_pred = rf.predict(X_test_scaled)
            
            # Calculate metrics
            fold_results['train_acc'].append(accuracy_score(y_train_smote, y_train_pred))
            fold_results['test_acc'].append(accuracy_score(y_test, y_test_pred))
            fold_results['train_f1'].append(f1_score(y_train_smote, y_train_pred))
            fold_results['test_f1'].append(f1_score(y_test, y_test_pred))
            fold_results['malware_recall'].append(recall_score(y_test, y_test_pred, pos_label=1))
            fold_results['malware_precision'].append(precision_score(y_test, y_test_pred, pos_label=1, zero_division=0))
        
        # Calculate statistics
        train_acc_mean = np.mean(fold_results['train_acc'])
        test_acc_mean = np.mean(fold_results['test_acc'])
        overfitting_gap = train_acc_mean - test_acc_mean
        
        print(f"   Train Accuracy: {train_acc_mean:.3f} ± {np.std(fold_results['train_acc']):.3f}")
        print(f"   Test Accuracy:  {test_acc_mean:.3f} ± {np.std(fold_results['test_acc']):.3f}")
        print(f"   Overfitting Gap: {overfitting_gap:.3f}")
        print(f"   Malware Recall: {np.mean(fold_results['malware_recall']):.3f} ± {np.std(fold_results['malware_recall']):.3f}")
        print(f"   Malware Precision: {np.mean(fold_results['malware_precision']):.3f} ± {np.std(fold_results['malware_precision']):.3f}")
        
        # Classify overfitting level
        if overfitting_gap < 0.02:
            status = "✅ Minimal"
        elif overfitting_gap < 0.05:
            status = "⚠️ Moderate"
        else:
            status = "❌ Severe"
        print(f"   Overfitting Status: {status}")
        
        results[config_name] = {
            'train_acc': fold_results['train_acc'],
            'test_acc': fold_results['test_acc'],
            'overfitting_gap': overfitting_gap,
            'malware_recall': fold_results['malware_recall'],
            'malware_precision': fold_results['malware_precision']
        }
    
    return results

def test_learning_curves(X, y):
    """Generate learning curves to detect overfitting"""
    
    print("\n📈 Test 2: Learning Curves Analysis")
    
    # Test with different RF configurations
    models = {
        'RF_Default': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'RF_Limited': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, 
                                           random_state=42, class_weight='balanced')
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"\n🔹 {model_name} Learning Curves")
        
        # Apply SMOTE to full dataset for learning curve
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_smote)
        
        # Generate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_scaled, y_smote, 
            train_sizes=train_sizes, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        ax = axes[idx]
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Accuracy')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Accuracy')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        ax.set_title(f'{model_name} Learning Curve')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Print final gap
        final_gap = train_mean[-1] - val_mean[-1]
        print(f"   Final Training Accuracy: {train_mean[-1]:.3f} ± {train_std[-1]:.3f}")
        print(f"   Final Validation Accuracy: {val_mean[-1]:.3f} ± {val_std[-1]:.3f}")
        print(f"   Final Gap: {final_gap:.3f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Learning curves saved to: {RESULTS_DIR}/learning_curves.png")

def test_validation_curves(X, y):
    """Test validation curves for key hyperparameters"""
    
    print("\n📈 Test 3: Validation Curves for Hyperparameters")
    
    # Apply SMOTE to full dataset
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_smote)
    
    # Test different hyperparameters
    param_tests = [
        {
            'param_name': 'n_estimators',
            'param_range': [10, 25, 50, 100, 200, 300],
            'title': 'Number of Trees'
        },
        {
            'param_name': 'max_depth',
            'param_range': [3, 5, 10, 15, 20, None],
            'title': 'Maximum Depth'
        },
        {
            'param_name': 'min_samples_split',
            'param_range': [2, 5, 10, 15, 20, 25],
            'title': 'Minimum Samples Split'
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, param_test in enumerate(param_tests):
        print(f"\n🔹 Testing {param_test['title']}")
        
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            X_scaled, y_smote,
            param_name=param_test['param_name'],
            param_range=param_test['param_range'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        ax = axes[idx]
        param_range_str = [str(p) for p in param_test['param_range']]
        x_pos = np.arange(len(param_range_str))
        
        ax.plot(x_pos, train_mean, 'o-', color='blue', label='Training')
        ax.fill_between(x_pos, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        ax.plot(x_pos, val_mean, 'o-', color='red', label='Validation')
        ax.fill_between(x_pos, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        ax.set_title(f'Validation Curve: {param_test["title"]}')
        ax.set_xlabel(param_test['param_name'])
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_range_str, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Find best parameter
        best_idx = np.argmax(val_mean)
        best_param = param_test['param_range'][best_idx]
        best_gap = train_mean[best_idx] - val_mean[best_idx]
        
        print(f"   Best {param_test['param_name']}: {best_param}")
        print(f"   Best Validation Accuracy: {val_mean[best_idx]:.3f}")
        print(f"   Overfitting Gap at Best: {best_gap:.3f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'validation_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Validation curves saved to: {RESULTS_DIR}/validation_curves.png")

def test_feature_importance_stability(X, y):
    """Test if feature importance is stable across folds"""
    
    print("\n🔍 Test 4: Feature Importance Stability")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_importances = []
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        
        # Train model
        rf.fit(X_train_scaled, y_train_smote)
        
        # Get feature importances
        importances = rf.feature_importances_
        all_importances.append(importances)
        
        print(f"   Fold {fold+1} - Top 3 features: {np.argsort(importances)[::-1][:3]}")
    
    # Calculate stability
    all_importances = np.array(all_importances)
    importance_std = np.std(all_importances, axis=0)
    importance_mean = np.mean(all_importances, axis=0)
    
    print(f"\n   Feature Importance Stability:")
    print(f"   Average STD across features: {np.mean(importance_std):.4f}")
    print(f"   Max STD: {np.max(importance_std):.4f}")
    print(f"   Most stable feature: {np.argmin(importance_std)} (STD: {np.min(importance_std):.4f})")
    print(f"   Least stable feature: {np.argmax(importance_std)} (STD: {np.max(importance_std):.4f})")
    
    # Plot feature importance stability
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(importance_mean))
    plt.bar(x_pos, importance_mean, yerr=importance_std, capsize=5, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Stability Across CV Folds')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance plot saved to: {RESULTS_DIR}/feature_importance_stability.png")

def create_summary_report(results):
    """Create a comprehensive summary report"""
    
    print("\n" + "="*80)
    print("📋 OVERFITTING ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n🔍 Configuration Comparison:")
    print(f"{'Model':<25} {'Test Acc':<12} {'Gap':<8} {'Status':<15} {'Malware Recall':<15}")
    print("-" * 80)
    
    for config_name, data in results.items():
        test_acc = np.mean(data['test_acc'])
        gap = data['overfitting_gap']
        malware_recall = np.mean(data['malware_recall'])
        
        if gap < 0.02:
            status = "✅ Minimal"
        elif gap < 0.05:
            status = "⚠️ Moderate"
        else:
            status = "❌ Severe"
        
        print(f"{config_name:<25} {test_acc:.3f}       {gap:.3f}    {status:<15} {malware_recall:.3f}")
    
    print("\n🎯 Key Findings:")
    
    # Find best configuration
    best_config = min(results.keys(), key=lambda x: results[x]['overfitting_gap'])
    best_gap = results[best_config]['overfitting_gap']
    
    print(f"   • Best Configuration: {best_config}")
    print(f"   • Lowest Overfitting Gap: {best_gap:.3f}")
    
    # Check if any configuration shows severe overfitting
    severe_overfitting = [k for k, v in results.items() if v['overfitting_gap'] > 0.05]
    if severe_overfitting:
        print(f"   • ⚠️ Configurations with severe overfitting: {', '.join(severe_overfitting)}")
    else:
        print(f"   • ✅ No severe overfitting detected in any configuration")
    
    # Check perfect performance
    perfect_configs = []
    for config_name, data in results.items():
        if np.mean(data['test_acc']) >= 0.999:
            perfect_configs.append(config_name)
    
    if perfect_configs:
        print(f"   • 🤔 Configurations with near-perfect performance: {', '.join(perfect_configs)}")
        print(f"     This could indicate overfitting or genuinely easy dataset")
    
    print(f"\n📊 Recommendation:")
    if best_gap < 0.02:
        print(f"   Random Forest appears to be generalizing well with minimal overfitting.")
        print(f"   The high performance may be due to:")
        print(f"   1. High-quality eigenspace embeddings that are highly discriminative")
        print(f"   2. Relatively simple decision boundary for this problem")
        print(f"   3. Effective feature engineering from graph-based opcode analysis")
    else:
        print(f"   Consider using regularization techniques or simpler models.")

def main():
    """Main overfitting analysis function"""
    print("=== Random Forest Overfitting Diagnostic ===")
    print("Comprehensive analysis to detect overfitting in Random Forest models")
    print("Using eigenspace embeddings from graph-based opcode analysis\n")
    
    # Load data
    X, y = load_data()
    if X is None:
        return
    
    # Run overfitting tests
    results = test_random_forest_overfitting(X, y)
    test_learning_curves(X, y)
    test_validation_curves(X, y)
    test_feature_importance_stability(X, y)
    
    # Create summary report
    create_summary_report(results)
    
    print(f"\n📁 All results saved to: {RESULTS_DIR}/")
    print("\nIf overfitting is detected, consider:")
    print("• Reducing model complexity (fewer trees, limited depth)")
    print("• Increasing min_samples_split and min_samples_leaf")
    print("• Using cross-validation for hyperparameter tuning")
    print("• Collecting more diverse training data")

if __name__ == "__main__":
    main()
