'''
Comprehensive Overfitting Analysis for IoT Malware Detection
Diagnoses and addresses potential overfitting, data leakage, and class imbalance issues
'''

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import os
from collections import Counter

# Create results directory
RESULTS_DIR = "Simple Overfitting Analysis Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Analysis results will be saved to: {RESULTS_DIR}/")

def load_data():
    """Load eigenspace embeddings and labels"""
    print("Loading eigenspace embeddings...")
    with open("X_graph_embeddings.pkl", "rb") as f:
        X_embeddings = pickle.load(f)
    
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
        labels = np.array(data["labels"])
        filenames = data.get("filenames", [f"sample_{i}" for i in range(len(labels))])
    
    print(f"Loaded {len(X_embeddings)} samples with {X_embeddings.shape[1]} features")
    print(f"Class distribution: Benign={np.sum(labels==0)}, Malware={np.sum(labels==1)}")
    
    return X_embeddings, labels, filenames

def analyze_data_leakage(X, labels, filenames):
    """Detect potential data leakage through sample similarity analysis"""
    print("\n=== Data Leakage Analysis ===")
    
    # Calculate similarity matrices
    cosine_sim = cosine_similarity(X)
    euclidean_dist = euclidean_distances(X)
    
    # Remove diagonal (self-similarity)
    np.fill_diagonal(cosine_sim, 0)
    np.fill_diagonal(euclidean_dist, np.inf)
    
    # Find highly similar pairs
    high_sim_threshold = 0.99
    similar_pairs = np.where(cosine_sim > high_sim_threshold)
    
    print(f"Sample pairs with >99% cosine similarity: {len(similar_pairs[0])}")
    
    # Analyze similarity within and across classes
    benign_indices = np.where(labels == 0)[0]
    malware_indices = np.where(labels == 1)[0]
    
    # Within-class similarities
    benign_sim = cosine_sim[np.ix_(benign_indices, benign_indices)]
    malware_sim = cosine_sim[np.ix_(malware_indices, malware_indices)]
    
    # Cross-class similarities
    cross_sim = cosine_sim[np.ix_(benign_indices, malware_indices)]
    
    print(f"Average within-benign similarity: {np.mean(benign_sim):.4f}")
    print(f"Average within-malware similarity: {np.mean(malware_sim):.4f}")
    print(f"Average cross-class similarity: {np.mean(cross_sim):.4f}")
    
    # Find potential duplicates/near-duplicates
    duplicate_pairs = []
    for i, j in zip(similar_pairs[0], similar_pairs[1]):
        if i < j:  # Avoid double counting
            duplicate_pairs.append((i, j, cosine_sim[i, j]))
    
    print(f"\nTop 10 most similar sample pairs:")
    sorted_pairs = sorted(duplicate_pairs, key=lambda x: x[2], reverse=True)[:10]
    for i, j, sim in sorted_pairs:
        print(f"  {filenames[i]} <-> {filenames[j]}: {sim:.6f} (labels: {labels[i]}, {labels[j]})")
    
    # Visualize similarity distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(benign_sim.flatten(), bins=50, alpha=0.7, label='Within Benign', density=True)
    plt.hist(malware_sim.flatten(), bins=50, alpha=0.7, label='Within Malware', density=True)
    plt.hist(cross_sim.flatten(), bins=50, alpha=0.7, label='Cross Class', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.xlim(0, 1)
    
    plt.subplot(1, 3, 2)
    similarity_heatmap = np.zeros((50, 50))
    indices = np.random.choice(len(X), size=50, replace=False)
    sim_subset = cosine_sim[np.ix_(indices, indices)]
    plt.imshow(sim_subset, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Sample Similarity Heatmap (Random 50)')
    
    plt.subplot(1, 3, 3)
    # Plot similarity vs class
    same_class_sim = []
    diff_class_sim = []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if labels[i] == labels[j]:
                same_class_sim.append(cosine_sim[i, j])
            else:
                diff_class_sim.append(cosine_sim[i, j])
    
    plt.hist(same_class_sim, bins=50, alpha=0.7, label='Same Class', density=True)
    plt.hist(diff_class_sim, bins=50, alpha=0.7, label='Different Class', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity by Class Relationship')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'data_leakage_analysis.png'), dpi=300)
    plt.close()
    
    return duplicate_pairs

def remove_duplicates(X, labels, filenames, similarity_threshold=0.995):
    """Remove near-duplicate samples"""
    print(f"\n=== Removing Near-Duplicates (threshold={similarity_threshold}) ===")
    
    cosine_sim = cosine_similarity(X)
    np.fill_diagonal(cosine_sim, 0)
    
    to_remove = set()
    removed_pairs = []
    
    for i in range(len(X)):
        if i in to_remove:
            continue
        for j in range(i+1, len(X)):
            if j in to_remove:
                continue
            if cosine_sim[i, j] > similarity_threshold:
                # Keep the one from the minority class if different classes
                if labels[i] != labels[j]:
                    if labels[i] == 1:  # Keep malware (minority)
                        to_remove.add(j)
                        removed_pairs.append((i, j, cosine_sim[i, j]))
                    else:
                        to_remove.add(i)
                        removed_pairs.append((j, i, cosine_sim[i, j]))
                        break
                else:
                    # Same class, remove the later one
                    to_remove.add(j)
                    removed_pairs.append((i, j, cosine_sim[i, j]))
    
    # Create deduplicated dataset
    keep_indices = [i for i in range(len(X)) if i not in to_remove]
    X_clean = X[keep_indices]
    labels_clean = labels[keep_indices]
    filenames_clean = [filenames[i] for i in keep_indices]
    
    print(f"Removed {len(to_remove)} duplicate samples")
    print(f"Remaining: {len(X_clean)} samples")
    print(f"New class distribution: Benign={np.sum(labels_clean==0)}, Malware={np.sum(labels_clean==1)}")
    
    return X_clean, labels_clean, filenames_clean, removed_pairs

def cross_validation_analysis(X, labels):
    """Perform robust cross-validation analysis"""
    print("\n=== Cross-Validation Analysis ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Multiple classifiers for robustness
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Stratified K-Fold to handle class imbalance
    cv_folds = [3, 5, 10]
    results = {}
    
    for clf_name, clf in classifiers.items():
        results[clf_name] = {}
        for k in cv_folds:
            if k <= np.sum(labels == 1):  # Ensure enough samples in minority class
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                scores = cross_val_score(clf, X_scaled, labels, cv=skf, scoring='accuracy')
                results[clf_name][f'{k}-fold'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
                print(f"{clf_name} {k}-fold CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Plot cross-validation results
    plt.figure(figsize=(12, 8))
    
    for i, (clf_name, clf_results) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        fold_names = list(clf_results.keys())
        means = [clf_results[fold]['mean'] for fold in fold_names]
        stds = [clf_results[fold]['std'] for fold in fold_names]
        
        plt.bar(fold_names, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title(f'{clf_name} Cross-Validation')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for j, (mean, std) in enumerate(zip(means, stds)):
            plt.text(j, mean + std + 0.02, f'{mean:.3f}', ha='center')
    
    # Overall comparison
    plt.subplot(2, 2, 4)
    all_means = []
    all_stds = []
    labels_plot = []
    
    for clf_name, clf_results in results.items():
        for fold_name, fold_result in clf_results.items():
            all_means.append(fold_result['mean'])
            all_stds.append(fold_result['std'])
            labels_plot.append(f'{clf_name}\n{fold_name}')
    
    plt.bar(range(len(all_means)), all_means, yerr=all_stds, capsize=3, alpha=0.7)
    plt.xticks(range(len(labels_plot)), labels_plot, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('All Cross-Validation Results')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cross_validation_analysis.png'), dpi=300)
    plt.close()
    
    return results

def feature_importance_analysis(X, labels):
    """Analyze feature importance and redundancy"""
    print("\n=== Feature Importance Analysis ===")
    
    # Use Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, labels)
    
    feature_importance = rf.feature_importances_
    
    # Find most and least important features
    top_features = np.argsort(feature_importance)[-20:]  # Top 20
    least_features = np.argsort(feature_importance)[:20]  # Bottom 20
    
    print(f"Top 20 most important features: {top_features}")
    print(f"Top 20 feature importance values: {feature_importance[top_features]}")
    print(f"Bottom 20 least important features: {least_features}")
    print(f"Bottom 20 feature importance values: {feature_importance[least_features]}")
    
    # Plot feature importance distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(feature_importance, bins=50, alpha=0.7)
    plt.xlabel('Feature Importance')
    plt.ylabel('Count')
    plt.title('Feature Importance Distribution')
    plt.axvline(np.mean(feature_importance), color='red', linestyle='--', label='Mean')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(sorted(feature_importance, reverse=True))
    plt.xlabel('Feature Rank')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Sorted)')
    plt.yscale('log')
    
    plt.subplot(1, 3, 3)
    # Correlation among top features
    top_feature_corr = np.corrcoef(X[:, top_features].T)
    plt.imshow(top_feature_corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.title('Top Features Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_analysis.png'), dpi=300)
    plt.close()
    
    return feature_importance

def dimensionality_analysis(X, labels):
    """Analyze data in reduced dimensions"""
    print("\n=== Dimensionality Analysis ===")
    
    # PCA Analysis
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Find number of components for 95% variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1
    
    print(f"Components for 95% variance: {n_95}")
    print(f"Components for 99% variance: {n_99}")
    
    # t-SNE for visualization
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X[:500])  # Subsample for speed
    labels_tsne = labels[:500]
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(cumvar[:100])  # First 100 components
    plt.axhline(0.95, color='red', linestyle='--', label='95%')
    plt.axhline(0.99, color='orange', linestyle='--', label='99%')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    colors = ['blue', 'red']
    class_names = ['Benign', 'Malware']
    for i, (color, name) in enumerate(zip(colors, class_names)):
        mask = labels_tsne == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name, alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization (Sample)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # PCA 2D visualization
    colors = ['blue', 'red']
    for i, (color, name) in enumerate(zip(colors, class_names)):
        mask = labels == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.title('PCA Visualization')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'dimensionality_analysis.png'), dpi=300)
    plt.close()
    
    return n_95, n_99

def class_balance_analysis(X, labels, filenames):
    """Analyze class balance and suggest improvements"""
    print("\n=== Class Balance Analysis ===")
    
    # Basic statistics
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    print(f"Class distribution:")
    for class_label, count in class_counts.items():
        class_name = 'Benign' if class_label == 0 else 'Malware'
        percentage = 100 * count / total_samples
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"Imbalance ratio (Benign:Malware): {imbalance_ratio:.1f}:1")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if imbalance_ratio > 10:
        print("⚠️  SEVERE CLASS IMBALANCE DETECTED")
        print("Recommendations:")
        print("1. Use stratified sampling in all train/test splits")
        print("2. Apply class weights in loss function")
        print("3. Use SMOTE or other oversampling techniques")
        print("4. Consider collecting more malware samples")
        print("5. Use metrics like F1-score, AUC instead of accuracy")
    
    # Suggest class weights for PyTorch
    weights = [total_samples / (2 * count) for count in class_counts.values()]
    print(f"Suggested class weights for PyTorch: {weights}")
    
    return imbalance_ratio, weights

def generate_summary_report(X_original, labels_original, X_clean, labels_clean, 
                          duplicate_pairs, cv_results, feature_importance, 
                          n_95, n_99, imbalance_ratio, weights, filenames_clean):
    """Generate comprehensive summary report"""
    print("\n=== Generating Summary Report ===")
    
    report = f"""
# Overfitting Analysis Report
## IoT Malware Detection Dataset Analysis

### Dataset Overview
- Original samples: {len(X_original)}
- Features per sample: {X_original.shape[1]}
- Original class distribution: Benign={np.sum(labels_original==0)}, Malware={np.sum(labels_original==1)}
- Class imbalance ratio: {imbalance_ratio:.1f}:1

### Data Quality Issues Detected

#### 1. Data Leakage / Near-Duplicates
- Found {len(duplicate_pairs)} highly similar sample pairs (>99.5% similarity)
- After deduplication: {len(X_clean)} samples remaining
- Clean class distribution: Benign={np.sum(labels_clean==0)}, Malware={np.sum(labels_clean==1)}

#### 2. Feature Analysis
- Dimensionality for 95% variance: {n_95} components
- Dimensionality for 99% variance: {n_99} components
- Top feature importance: {np.max(feature_importance):.6f}
- Mean feature importance: {np.mean(feature_importance):.6f}

#### 3. Cross-Validation Results (Clean Data)
"""
    
    for clf_name, clf_results in cv_results.items():
        report += f"\n##### {clf_name}\n"
        for fold_name, fold_result in clf_results.items():
            report += f"- {fold_name}: {fold_result['mean']:.4f} ± {fold_result['std']:.4f}\n"
    
    report += f"""
### Recommendations for Addressing Overfitting

#### 1. Data Quality
- ✅ Remove {len(duplicate_pairs)} near-duplicate samples
- 🔄 Use the cleaned dataset with {len(X_clean)} samples
- 🔍 Investigate data collection process to prevent future duplicates

#### 2. Model Training
- 🎯 Use stratified cross-validation (not simple train/test split)
- ⚖️ Apply class weights: {weights}
- 📏 Use multiple evaluation metrics (F1, AUC, not just accuracy)
- 🔀 Apply data augmentation or SMOTE for minority class

#### 3. Model Architecture
- 🏗️ Reduce model complexity (fewer parameters)
- 🛡️ Increase regularization (dropout, weight decay)
- 📊 Use early stopping based on validation loss
- 🎲 Ensemble multiple models with different random seeds

#### 4. Evaluation Protocol
- 📋 Use nested cross-validation for hyperparameter tuning
- 🔄 Report confidence intervals across multiple runs
- 📈 Plot learning curves to detect overfitting
- 🎯 Use hold-out test set from different source if possible

### Files Generated
- data_leakage_analysis.png
- cross_validation_analysis.png  
- feature_importance_analysis.png
- dimensionality_analysis.png
- cleaned_dataset.pkl (recommended for future use)
"""
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, 'overfitting_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to: {report_path}")
    
    # Save cleaned dataset
    cleaned_data = {
        'embeddings': X_clean,
        'labels': labels_clean.tolist(),
        'filenames': filenames_clean
    }
    
    with open(os.path.join(RESULTS_DIR, 'cleaned_dataset.pkl'), 'wb') as f:
        pickle.dump(cleaned_data, f)
    
    print(f"Cleaned dataset saved to: {RESULTS_DIR}/cleaned_dataset.pkl")

def main():
    print("=== Comprehensive Overfitting Analysis ===")
    print("This analysis will identify and address overfitting issues in the IoT malware dataset")
    
    # Load data
    X, labels, filenames = load_data()
    
    # 1. Data leakage analysis
    duplicate_pairs = analyze_data_leakage(X, labels, filenames)
    
    # 2. Remove duplicates
    X_clean, labels_clean, filenames_clean, removed_pairs = remove_duplicates(X, labels, filenames)
    
    # 3. Cross-validation on clean data
    cv_results = cross_validation_analysis(X_clean, labels_clean)
    
    # 4. Feature analysis
    feature_importance = feature_importance_analysis(X_clean, labels_clean)
    
    # 5. Dimensionality analysis
    n_95, n_99 = dimensionality_analysis(X_clean, labels_clean)
    
    # 6. Class balance analysis
    imbalance_ratio, weights = class_balance_analysis(X_clean, labels_clean, filenames_clean)
    
    # 7. Generate summary report
    generate_summary_report(X, labels, X_clean, labels_clean, removed_pairs, 
                          cv_results, feature_importance, n_95, n_99, 
                          imbalance_ratio, weights, filenames_clean)
    
    print(f"\n=== Analysis Complete ===")
    print(f"All results saved to: {RESULTS_DIR}/")
    print(f"\nKey findings:")
    print(f"- Removed {len(removed_pairs)} near-duplicate samples")
    print(f"- Class imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"- Effective dimensionality: {n_95} components (95% variance)")
    print(f"- Cleaned dataset ready for robust evaluation")

if __name__ == "__main__":
    main()
