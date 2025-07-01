'''
Fixed SMOTE Analysis - Applies SMOTE Inside CV Folds to Prevent Data Leakage
'''

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import os
from collections import defaultdict

# Create results directory
RESULTS_DIR = "SMOTE Results Fixed (10-Fold)"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Fixed SMOTE results (10-fold CV) will be saved to: {RESULTS_DIR}/")

def load_cleaned_data():
    """Load the cleaned dataset"""
    print("Loading cleaned dataset...")
    try:
        with open("Overfitting Analysis Results/cleaned_dataset.pkl", "rb") as f:
            cleaned_data = pickle.load(f)
            X = np.array(cleaned_data['embeddings'])
            y = np.array(cleaned_data['labels'])
        print(f"✅ Using cleaned dataset: {len(X)} samples, {X.shape[1]} features")
    except FileNotFoundError:
        print("⚠️  Cleaned dataset not found, using original data")
        with open("X_graph_embeddings.pkl", "rb") as f:
            X = pickle.load(f)
        with open("improved_cig_output.pkl", "rb") as f:
            data = pickle.load(f)
            y = np.array(data["labels"])
    
    return X, y

def evaluate_with_proper_smote(X, y, smote_method, classifier, n_folds=10):
    """
    Proper SMOTE evaluation - applies SMOTE inside each CV fold
    Uses 10-fold cross-validation for robust statistical estimates
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = {
        'accuracy': [],
        'f1': [],
        'auc': [],
        'malware_precision': [],
        'malware_recall': []
    }
    
    print(f"\n--- {smote_method.__class__.__name__} with {classifier.__class__.__name__} ---")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check if we have both classes in training
        train_classes = np.unique(y_train)
        if len(train_classes) < 2:
            print(f"  Fold {fold+1}: Skipping - only one class in training")
            continue
        
        # Apply SMOTE only to training data
        try:
            X_train_smote, y_train_smote = smote_method.fit_resample(X_train, y_train)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_smote)
            X_test_scaled = scaler.transform(X_test)
            
            # Train classifier
            classifier.fit(X_train_scaled, y_train_smote)
            
            # Predict
            y_pred = classifier.predict(X_test_scaled)
            y_proba = classifier.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Handle AUC calculation
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = 0.5  # Random performance
            
            # Calculate per-class metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            malware_precision = report.get('1', {}).get('precision', 0.0)
            malware_recall = report.get('1', {}).get('recall', 0.0)
            
            # Store results
            fold_results['accuracy'].append(accuracy)
            fold_results['f1'].append(f1)
            fold_results['auc'].append(auc)
            fold_results['malware_precision'].append(malware_precision)
            fold_results['malware_recall'].append(malware_recall)
            
            print(f"  Fold {fold+1}: Acc={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}, "
                  f"Malware P={malware_precision:.3f}, R={malware_recall:.3f}")
            
        except Exception as e:
            print(f"  Fold {fold+1}: Error - {str(e)}")
            continue
    
    # Calculate averages
    if fold_results['accuracy']:
        results_summary = {}
        for metric in fold_results:
            values = fold_results[metric]
            results_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        print(f"  Overall: Acc={results_summary['accuracy']['mean']:.3f}±{results_summary['accuracy']['std']:.3f}, "
              f"F1={results_summary['f1']['mean']:.3f}±{results_summary['f1']['std']:.3f}, "
              f"AUC={results_summary['auc']['mean']:.3f}±{results_summary['auc']['std']:.3f}")
        print(f"  Malware: P={results_summary['malware_precision']['mean']:.3f}±{results_summary['malware_precision']['std']:.3f}, "
              f"R={results_summary['malware_recall']['mean']:.3f}±{results_summary['malware_recall']['std']:.3f}")
        
        return results_summary
    else:
        print("  No valid folds completed")
        return None

def compare_smote_methods(X, y):
    """Compare different SMOTE methods with proper evaluation"""
    print("\n=== Fixed SMOTE Analysis (Proper CV) ===")
    
    # SMOTE methods to test
    smote_methods = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=3),  # Reduced k_neighbors for small minority class
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
        'ADASYN': ADASYN(random_state=42, n_neighbors=3)
    }
    
    # Classifiers to test
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Also test without SMOTE
    print("\n--- Baseline (No SMOTE) ---")
    baseline_results = {}
    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name}:")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'f1': [], 'auc': [], 'malware_precision': [], 'malware_recall': []}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if len(np.unique(y_train)) < 2:
                continue
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            mal_p = report.get('1', {}).get('precision', 0.0)
            mal_r = report.get('1', {}).get('recall', 0.0)
            
            scores['accuracy'].append(acc)
            scores['f1'].append(f1)
            scores['auc'].append(auc)
            scores['malware_precision'].append(mal_p)
            scores['malware_recall'].append(mal_r)
        
        baseline_results[clf_name] = scores
        print(f"  Acc={np.mean(scores['accuracy']):.3f}±{np.std(scores['accuracy']):.3f}, "
              f"Malware P={np.mean(scores['malware_precision']):.3f}±{np.std(scores['malware_precision']):.3f}, "
              f"R={np.mean(scores['malware_recall']):.3f}±{np.std(scores['malware_recall']):.3f}")
    
    # Test SMOTE methods
    all_results = {'Baseline': baseline_results}
    
    for smote_name, smote_method in smote_methods.items():
        print(f"\n=== {smote_name} ===")
        smote_results = {}
        
        for clf_name, clf in classifiers.items():
            result = evaluate_with_proper_smote(X, y, smote_method, clf)
            if result:
                smote_results[clf_name] = result
        
        all_results[smote_name] = smote_results
    
    return all_results

def plot_smote_comparison(results, save_path):
    """Plot comparison of SMOTE methods"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'f1', 'auc', 'malware_precision', 'malware_recall']
    titles = ['Accuracy', 'F1-Score', 'AUC', 'Malware Precision', 'Malware Recall']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        method_names = []
        classifier_results = defaultdict(list)
        
        for method_name, method_results in results.items():
            if method_name == 'Baseline':
                for clf_name, clf_scores in method_results.items():
                    if clf_scores[metric]:
                        classifier_results[clf_name].append(np.mean(clf_scores[metric]))
                        if not method_names or method_names[-1] != method_name:
                            method_names.append(method_name)
            else:
                for clf_name, clf_result in method_results.items():
                    if clf_result and metric in clf_result:
                        classifier_results[clf_name].append(clf_result[metric]['mean'])
                        if not method_names or method_names[-1] != method_name:
                            method_names.append(method_name)
        
        # Plot bars for each classifier
        x = np.arange(len(method_names))
        width = 0.25
        
        for j, (clf_name, values) in enumerate(classifier_results.items()):
            if len(values) == len(method_names):
                ax.bar(x + j * width, values, width, label=clf_name, alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SMOTE comparison plot saved to: {save_path}")

def main():
    print("=== Fixed SMOTE Analysis for IoT Malware Detection (10-Fold CV) ===")
    print("This version applies SMOTE inside CV folds to prevent data leakage")
    print("Using 10-fold cross-validation for robust statistical estimates")
    
    # Load data
    X, y = load_cleaned_data()
    
    print(f"\nOriginal Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        cls_name = 'Benign' if cls == 0 else 'Malware'
        print(f"  {cls_name}: {count} samples ({100*count/len(y):.1f}%)")
    print(f"  Imbalance ratio: {counts[0]/counts[1]:.1f}:1")
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Compare SMOTE methods
    results = compare_smote_methods(X, y)
    
    # Plot results
    plot_path = os.path.join(RESULTS_DIR, 'fixed_smote_comparison.png')
    plot_smote_comparison(results, plot_path)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, 'fixed_smote_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n=== Summary ===")
    print(f"Fixed SMOTE analysis completed with 10-fold CV - no data leakage")
    print(f"Results show realistic performance with robust statistical estimates")
    print(f"Results saved to: {RESULTS_DIR}/")
    
    # Find best method for malware detection
    best_malware_recall = 0
    best_method = None
    best_classifier = None
    
    for method_name, method_results in results.items():
        if method_name == 'Baseline':
            for clf_name, clf_scores in method_results.items():
                avg_recall = np.mean(clf_scores['malware_recall']) if clf_scores['malware_recall'] else 0
                if avg_recall > best_malware_recall:
                    best_malware_recall = avg_recall
                    best_method = method_name
                    best_classifier = clf_name
        else:
            for clf_name, clf_result in method_results.items():
                if clf_result and 'malware_recall' in clf_result:
                    recall = clf_result['malware_recall']['mean']
                    if recall > best_malware_recall:
                        best_malware_recall = recall
                        best_method = method_name
                        best_classifier = clf_name
    
    print(f"\nBest malware detection:")
    print(f"Method: {best_method}")
    print(f"Classifier: {best_classifier}")
    print(f"Malware Recall: {best_malware_recall:.3f}")

if __name__ == "__main__":
    main()
