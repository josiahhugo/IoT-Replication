'''
Quick Simple Models Analysis for IoT Malware Detection
Fast evaluation of the most effective models only
'''

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from alive_progress import alive_bar
import warnings
warnings.filterwarnings('ignore')
import os

# Create results directory
RESULTS_DIR = "Quick Simple Models Analysis Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}/")

def load_data():
    """Load eigenspace embeddings and labels"""
    try:
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
            with open('X_graph_embeddings.pkl', 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    X = data.get('embeddings', data.get('X', None))
                else:
                    X = data
            print(f"✅ Loaded regular eigenspace embeddings")
        
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

def evaluate_model_quick(model, X, y, model_name, use_smote=False):
    """Quick evaluation with 3-fold CV"""
    
    print(f"🔄 {model_name}...")
    
    # Define scoring
    scoring = {
        'accuracy': 'accuracy',
        'malware_recall': make_scorer(custom_scorer_malware_recall),
        'benign_recall': make_scorer(custom_scorer_benign_recall),
        'malware_f1': make_scorer(custom_scorer_malware_f1),
        'benign_f1': make_scorer(custom_scorer_benign_f1),
        'malware_precision': make_scorer(custom_scorer_malware_precision),
        'benign_precision': make_scorer(custom_scorer_benign_precision),
        'roc_auc': 'roc_auc'
    }
    
    # Create pipeline
    if use_smote:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
    else:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    
    # Quick 3-fold CV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, 
                               return_train_score=True, n_jobs=-1)
    
    # Calculate statistics
    results = {}
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[metric] = {
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'train_mean': np.mean(train_scores),
            'overfitting_gap': np.mean(train_scores) - np.mean(test_scores)
        }
    
    return results

def print_results_table(all_results):
    """Print results in a clean table format"""
    
    print(f"\n{'='*100}")
    print("📊 SIMPLE MODELS ANALYSIS RESULTS")
    print(f"{'='*100}")
    
    # Create summary table
    rows = []
    for model_name, results in all_results.items():
        row = {
            'Model': model_name,
            'Overall_Accuracy': f"{results['accuracy']['test_mean']:.3f} ± {results['accuracy']['test_std']:.3f}",
            'Malware_Precision': f"{results['malware_precision']['test_mean']:.3f} ± {results['malware_precision']['test_std']:.3f}",
            'Malware_Recall': f"{results['malware_recall']['test_mean']:.3f} ± {results['malware_recall']['test_std']:.3f}",
            'Malware_F1': f"{results['malware_f1']['test_mean']:.3f} ± {results['malware_f1']['test_std']:.3f}",
            'Benign_Precision': f"{results['benign_precision']['test_mean']:.3f} ± {results['benign_precision']['test_std']:.3f}",
            'Benign_Recall': f"{results['benign_recall']['test_mean']:.3f} ± {results['benign_recall']['test_std']:.3f}",
            'Benign_F1': f"{results['benign_f1']['test_mean']:.3f} ± {results['benign_f1']['test_std']:.3f}",
            'ROC_AUC': f"{results['roc_auc']['test_mean']:.3f} ± {results['roc_auc']['test_std']:.3f}",
            'Overfitting_Gap': f"{results['accuracy']['overfitting_gap']:.3f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'quick_simple_models_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")

def print_detailed_breakdown(all_results):
    """Print detailed per-model breakdown"""
    
    print(f"\n{'='*80}")
    print("📋 DETAILED BREAKDOWN BY MODEL")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        print(f"\n🔹 {model_name}")
        print(f"{'─' * (len(model_name) + 3)}")
        
        print(f"📈 OVERALL METRICS:")
        print(f"   Accuracy:      {results['accuracy']['test_mean']:.3f} ± {results['accuracy']['test_std']:.3f}")
        print(f"   ROC AUC:       {results['roc_auc']['test_mean']:.3f} ± {results['roc_auc']['test_std']:.3f}")
        print(f"   Overfitting:   {results['accuracy']['overfitting_gap']:.3f}")
        
        print(f"\n🦠 MALWARE METRICS:")
        print(f"   Precision:     {results['malware_precision']['test_mean']:.3f} ± {results['malware_precision']['test_std']:.3f}")
        print(f"   Recall:        {results['malware_recall']['test_mean']:.3f} ± {results['malware_recall']['test_std']:.3f}")
        print(f"   F1-Score:      {results['malware_f1']['test_mean']:.3f} ± {results['malware_f1']['test_std']:.3f}")
        
        print(f"\n✅ BENIGN METRICS:")
        print(f"   Precision:     {results['benign_precision']['test_mean']:.3f} ± {results['benign_precision']['test_std']:.3f}")
        print(f"   Recall:        {results['benign_recall']['test_mean']:.3f} ± {results['benign_recall']['test_std']:.3f}")
        print(f"   F1-Score:      {results['benign_f1']['test_mean']:.3f} ± {results['benign_f1']['test_std']:.3f}")

def main():
    """Quick analysis of top models"""
    print("=== Quick Simple Models Analysis ===")
    print("Fast evaluation of Logistic Regression and Random Forest")
    print("With detailed per-class metrics for benign and malware detection\n")
    
    # Load data
    X, y = load_data()
    if X is None:
        return
    
    # Create fast models
    models = {
        'Logistic Regression (Balanced)': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced', solver='liblinear'
        ),
        'Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=50, random_state=42, class_weight='balanced', 
            n_jobs=-1, max_depth=10
        ),
        'Logistic Regression (No Balance)': LogisticRegression(
            random_state=42, max_iter=1000, solver='liblinear'
        )
    }
    
    # Evaluate models
    all_results = {}
    
    print(f"\n🔄 Evaluating {len(models)} models with 3-fold CV...")
    with alive_bar(len(models), title="Models") as bar:
        for model_name, model in models.items():
            results = evaluate_model_quick(model, X, y, model_name)
            all_results[model_name] = results
            bar()
    
    # Test with SMOTE
    print(f"\n🔄 Testing with SMOTE...")
    smote_models = ['Logistic Regression (Balanced)', 'Random Forest (Balanced)']
    
    with alive_bar(len(smote_models), title="SMOTE") as bar:
        for model_name in smote_models:
            model = models[model_name]
            smote_name = f"{model_name} + SMOTE"
            results = evaluate_model_quick(model, X, y, smote_name, use_smote=True)
            all_results[smote_name] = results
            bar()
    
    # Print results
    print_results_table(all_results)
    print_detailed_breakdown(all_results)
    
    # Find best performers
    print(f"\n{'='*80}")
    print("🏆 TOP PERFORMERS")
    print(f"{'='*80}")
    
    # Best malware recall
    best_malware_recall = max(all_results.items(), 
                             key=lambda x: x[1]['malware_recall']['test_mean'])
    print(f"🥇 Best Malware Recall: {best_malware_recall[0]}")
    print(f"   Score: {best_malware_recall[1]['malware_recall']['test_mean']:.3f}")
    
    # Best overall accuracy
    best_accuracy = max(all_results.items(), 
                       key=lambda x: x[1]['accuracy']['test_mean'])
    print(f"🥇 Best Overall Accuracy: {best_accuracy[0]}")
    print(f"   Score: {best_accuracy[1]['accuracy']['test_mean']:.3f}")
    
    # Best F1 for malware
    best_malware_f1 = max(all_results.items(), 
                         key=lambda x: x[1]['malware_f1']['test_mean'])
    print(f"🥇 Best Malware F1: {best_malware_f1[0]}")
    print(f"   Score: {best_malware_f1[1]['malware_f1']['test_mean']:.3f}")
    
    print(f"\n📁 Results saved to: {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
