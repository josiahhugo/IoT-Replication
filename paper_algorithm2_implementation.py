'''
CORRECTED Implementation of Algorithm 2: Junk Code Insertion Procedure
Following the exact algorithm from the paper with proper error handling
'''
import pickle
import numpy as np
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def algorithm2_junk_insertion(adjacency_matrix, junk_percentage_k, sample_id=0):
    """
    CORRECTED Implementation of Algorithm 2 from the paper
    
    Algorithm 2: Junk Code Insertion Procedure
    Input: Trained Classifier D, Test Samples S, Junk Code Percentage k
    Output: Predicted Class for Test Samples P
    """
    # Step 3: W = Compute the CFG of sample (adjacency matrix)
    W = adjacency_matrix.copy()
    
    # Step 4: R = select k% of W's indices randomly (Allow duplicate indices)
    total_elements = W.size
    num_to_select = max(1, int(total_elements * junk_percentage_k / 100))
    
    # Use deterministic seed based on sample and junk percentage for reproducibility
    np.random.seed(42 + int(junk_percentage_k * 10) + sample_id)
    selected_indices = np.random.randint(0, total_elements, size=num_to_select)
    
    # Step 5-7: For each index in R do: W_index = W_index + 1
    for idx in selected_indices:
        row, col = np.unravel_index(idx, W.shape)
        W[row, col] += 1
    
    # Step 8: Normalize W
    W_normalized = normalize_adjacency_matrix(W)
    
    # Step 9-10: e1, e2 = 1st and 2nd eigenvectors of W
    # l1, l2 = 1st and 2nd eigenvalues of W
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(W_normalized)
        
        # Sort eigenvalues and eigenvectors by eigenvalue magnitude (descending)
        idx_sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]
        
        # Extract first and second eigenvectors and eigenvalues
        e1 = eigenvectors[:, 0] if eigenvectors.shape[1] > 0 else np.zeros(W.shape[0])
        e2 = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(W.shape[0])
        l1 = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
        l2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        # Handle NaN/Inf values
        e1 = np.nan_to_num(e1, nan=0.0, posinf=1.0, neginf=-1.0)
        e2 = np.nan_to_num(e2, nan=0.0, posinf=1.0, neginf=-1.0)
        l1 = float(np.nan_to_num(l1, nan=0.0, posinf=1.0, neginf=-1.0))
        l2 = float(np.nan_to_num(l2, nan=0.0, posinf=1.0, neginf=-1.0))
        
    except np.linalg.LinAlgError as e:
        print(f"⚠️  LinAlg error for {junk_percentage_k}% junk: {e}")
        # Fallback to zero features
        e1 = np.zeros(W.shape[0])
        e2 = np.zeros(W.shape[0])
        l1, l2 = 0.0, 0.0
    
    # Step 11: P = P ∪ D(e1, e2, l1, l2) - combine features for classification
    modified_features = np.concatenate([e1, e2, [l1, l2]])
    
    return modified_features, W_normalized

def normalize_adjacency_matrix(matrix):
    """
    Robust normalization of adjacency matrix
    """
    # Ensure matrix is non-negative
    matrix = np.abs(matrix)
    
    # Add small regularization to avoid singular matrices
    matrix += np.eye(matrix.shape[0]) * 1e-6
    
    # Row normalization (standard for adjacency matrices)
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = matrix / row_sums[:, np.newaxis]
    
    return normalized

def load_clean_data():
    """Load and deduplicate data exactly like CNN training"""
    print("📂 Loading cleaned data...")
    
    try:
        with open("improved_cig_output.pkl", "rb") as f:
            data = pickle.load(f)
        with open("X_graph_embeddings.pkl", "rb") as f:
            X_embeddings = pickle.load(f)
        
        X = np.array(X_embeddings, dtype=np.float32)
        y = np.array(data["labels"], dtype=np.float32)
        
        # Remove duplicates - EXACTLY like CNN training
        unique_indices = []
        seen_hashes = set()
        for i, sample in enumerate(X):
            sample_hash = hashlib.md5(sample.tobytes()).hexdigest()
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unique_indices.append(i)
        
        X_clean = X[unique_indices]
        y_clean = y[unique_indices]
        
        print(f"✅ Data loaded: {len(X_clean)} clean samples")
        print(f"   Distribution: {Counter(y_clean)}")
        
        return X_clean, y_clean
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def test_corrected_algorithm2():
    """
    Test the corrected Algorithm 2 implementation
    """
    print("🧪 TESTING CORRECTED ALGORITHM 2: JUNK CODE INSERTION")
    print("=" * 60)
    print("Following the exact paper algorithm with proper error handling")
    print("=" * 60)
    
    # Load cleaned data
    X_clean, y_clean = load_clean_data()
    if X_clean is None:
        return None
    
    # Create train/test split
    test_size = 200
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    indices = list(range(len(y_clean)))
    
    for train_indices, test_indices in sss.split(indices, y_clean):
        print(f"📊 Data split:")
        print(f"   Training samples: {len(train_indices)}")
        print(f"   Test samples: {len(test_indices)}")
        print(f"   Train distribution: {Counter(y_clean[train_indices])}")
        print(f"   Test distribution: {Counter(y_clean[test_indices])}")
        
        # Determine matrix dimensions
        embedding_dim = int(np.sqrt(X_clean.shape[1]))
        max_dim = min(embedding_dim, 31)  # Cap at 31 for manageable size
        
        print(f"   Using {max_dim}x{max_dim} adjacency matrices")
        
        # Create adjacency matrices from embeddings
        def create_adjacency_matrix(embedding):
            matrix_flat = embedding[:max_dim*max_dim]
            matrix = matrix_flat.reshape(max_dim, max_dim)
            # Make symmetric and positive
            matrix = (matrix + matrix.T) / 2
            matrix = np.abs(matrix)
            return matrix
        
        # Extract training features (Step 1: Train Classifier D)
        print(f"\n🔧 Training Classifier D...")
        
        train_features = []
        failed_extractions = 0
        
        for i, idx in enumerate(train_indices):
            try:
                matrix = create_adjacency_matrix(X_clean[idx])
                features, _ = algorithm2_junk_insertion(matrix, 0, sample_id=i)  # 0% junk for baseline
                train_features.append(features)
            except Exception as e:
                print(f"⚠️  Failed to extract training features for sample {i}: {e}")
                # Use zero features as fallback
                train_features.append(np.zeros(max_dim * 2 + 2))
                failed_extractions += 1
        
        train_features = np.array(train_features)
        train_labels = y_clean[train_indices]
        
        if failed_extractions > 0:
            print(f"   ⚠️  {failed_extractions} training samples used fallback features")
        
        # Train classifier D
        classifier_D = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        classifier_D.fit(train_features, train_labels)
        
        baseline_train_accuracy = classifier_D.score(train_features, train_labels)
        print(f"   Classifier D training accuracy: {baseline_train_accuracy:.3f}")
        
        # Test Algorithm 2 with different junk percentages
        print(f"\n🧪 Testing Algorithm 2 Junk Insertion Procedure...")
        
        junk_percentages = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        results = {
            'junk_percentages': junk_percentages,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f_measure': [],
            'feature_failures': []
        }
        
        for k in junk_percentages:
            print(f"\n📈 Testing k = {k}% junk insertion...")
            
            # Apply Algorithm 2 to test samples
            test_features = []
            extraction_failures = 0
            
            for i, idx in enumerate(test_indices):
                try:
                    matrix = create_adjacency_matrix(X_clean[idx])
                    # Apply Algorithm 2
                    features, _ = algorithm2_junk_insertion(matrix, k, sample_id=i)
                    test_features.append(features)
                except Exception as e:
                    print(f"⚠️  Failed to extract test features for sample {i}: {e}")
                    # Use zero features as fallback
                    test_features.append(np.zeros(max_dim * 2 + 2))
                    extraction_failures += 1
            
            test_features = np.array(test_features)
            test_labels = y_clean[test_indices]
            
            # Ensure feature compatibility
            if test_features.shape[1] != train_features.shape[1]:
                min_features = min(test_features.shape[1], train_features.shape[1])
                test_features = test_features[:, :min_features]
                
                if test_features.shape[1] < train_features.shape[1]:
                    padding = np.zeros((test_features.shape[0], 
                                     train_features.shape[1] - test_features.shape[1]))
                    test_features = np.hstack([test_features, padding])
            
            # Step 12: Get predictions using classifier D
            try:
                predictions = classifier_D.predict(test_features)
                
                # Calculate metrics
                accuracy = accuracy_score(test_labels, predictions) * 100
                precision = precision_score(test_labels, predictions, average='macro', zero_division=0) * 100
                recall = recall_score(test_labels, predictions, average='macro', zero_division=0) * 100
                f_measure = f1_score(test_labels, predictions, average='macro', zero_division=0) * 100
                
                # Store results
                results['accuracy'].append(accuracy)
                results['precision'].append(precision)
                results['recall'].append(recall)
                results['f_measure'].append(f_measure)
                results['feature_failures'].append(extraction_failures)
                
                # Debug information
                pred_counts = Counter(predictions)
                true_counts = Counter(test_labels)
                
                print(f"   📊 Algorithm 2 Results:")
                print(f"      True labels: {dict(true_counts)}")
                print(f"      Predictions: {dict(pred_counts)}")
                print(f"      Accuracy: {accuracy:.1f}%")
                print(f"      Precision: {precision:.1f}%")
                print(f"      Recall: {recall:.1f}%")
                print(f"      F-measure: {f_measure:.1f}%")
                print(f"      Feature failures: {extraction_failures}/{len(test_indices)}")
                
            except Exception as e:
                print(f"❌ Error with {k}% junk: {e}")
                results['accuracy'].append(0)
                results['precision'].append(0)
                results['recall'].append(0)
                results['f_measure'].append(0)
                results['feature_failures'].append(len(test_indices))
        
        # Analysis and visualization
        print(f"\n📋 CORRECTED ALGORITHM 2 SUMMARY:")
        baseline_accuracy = results['accuracy'][0]
        final_accuracy = results['accuracy'][-1]
        total_drop = baseline_accuracy - final_accuracy
        
        print(f"   🎯 Baseline accuracy: {baseline_accuracy:.1f}%")
        print(f"   📉 Final accuracy: {final_accuracy:.1f}%")
        print(f"   📊 Total performance drop: {total_drop:.1f}%")
        print(f"   🛡️  Algorithm resilience: {(final_accuracy/baseline_accuracy)*100:.1f}%")
        
        # Performance variance analysis
        performance_variance = np.var(results['accuracy'])
        print(f"   📊 Performance variance: {performance_variance:.2f}")
        
        if performance_variance > 10:
            print(f"   ✅ HIGH SENSITIVITY: Algorithm responds strongly to junk insertion")
        elif performance_variance > 1:
            print(f"   ✅ MODERATE SENSITIVITY: Algorithm shows measurable response")
        else:
            print(f"   ⚠️  LOW SENSITIVITY: Algorithm may be too resilient or needs debugging")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.plot(results['junk_percentages'], results['accuracy'], 'o-', 
                linewidth=3, markersize=10, label='Accuracy', color='blue')
        plt.plot(results['junk_percentages'], results['precision'], 's-', 
                linewidth=3, markersize=10, label='Precision', color='green')
        plt.plot(results['junk_percentages'], results['recall'], '^-', 
                linewidth=3, markersize=10, label='Recall', color='red')
        plt.plot(results['junk_percentages'], results['f_measure'], 'D-', 
                linewidth=3, markersize=10, label='F-Measure', color='purple')
        
        plt.xlabel('Junk Code Percentage k (%)', fontsize=14)
        plt.ylabel('Performance (%)', fontsize=14)
        plt.title('Algorithm 2: Junk Code Insertion Procedure\n(Corrected Implementation)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 40)
        plt.ylim(0, 100)
        
        # Add performance drop annotations
        for i, (x, y) in enumerate(zip(results['junk_percentages'], results['accuracy'])):
            if x > 0:
                drop = baseline_accuracy - y
                if abs(drop) > 0.5:
                    plt.annotate(f'{drop:+.1f}%', 
                               xy=(x, y), xytext=(0, 10), 
                               textcoords='offset points',
                               fontsize=10, ha='center', alpha=0.8)
        
        plt.tight_layout()
        
        # Save results - CREATE DIRECTORY FIRST
        import os
        os.makedirs("Algorithm2 Results", exist_ok=True)
        
        plt.savefig("Algorithm2 Results/corrected_algorithm2_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        with open("Algorithm2 Results/corrected_algorithm2_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Corrected Algorithm 2 results saved:")
        print(f"   📄 Algorithm2 Results/corrected_algorithm2_results.json")
        print(f"   📊 Algorithm2 Results/corrected_algorithm2_results.png")
        
        return results

if __name__ == "__main__":
    test_corrected_algorithm2()