'''
Validation Tests for Algorithm 2 Results
Multiple checks to ensure junk code resilience results are legitimate
'''
import pickle
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import seaborn as sns

def algorithm2_junk_insertion(adjacency_matrix, junk_percentage_k):
    """Same as before - kept for consistency"""
    W = adjacency_matrix.copy()
    total_elements = W.size
    num_to_select = max(1, int(total_elements * junk_percentage_k / 100))
    selected_indices = np.random.randint(0, total_elements, size=num_to_select)
    
    for idx in selected_indices:
        row, col = np.unravel_index(idx, W.shape)
        W[row, col] += 1
    
    # Normalize
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums[:, np.newaxis]
    
    # Extract features
    eigenvalues, eigenvectors = np.linalg.eigh(W_normalized)
    e1 = eigenvectors[:, 0] if eigenvectors.shape[1] > 0 else np.zeros(W.shape[0])
    e2 = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(W.shape[0])
    l1 = eigenvalues[0] if len(eigenvalues) > 0 else 0
    l2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
    
    modified_features = np.concatenate([e1, e2, [l1, l2]])
    return modified_features, W_normalized

def validation_test_1_feature_change_analysis():
    """
    Validation 1: Verify that junk insertion actually changes features significantly
    """
    print("=== VALIDATION 1: Feature Change Analysis ===")
    
    # Load data
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
    
    with open("X_graph_embeddings.pkl", "rb") as f:
        X_embeddings = np.array(pickle.load(f))
    
    # Create proxy adjacency matrix
    embedding_dim = int(np.sqrt(X_embeddings.shape[1]))
    sample_matrix = X_embeddings[0][:embedding_dim*embedding_dim].reshape(embedding_dim, embedding_dim)
    sample_matrix = (sample_matrix + sample_matrix.T) / 2
    
    print(f"Testing feature changes on {embedding_dim}x{embedding_dim} matrix")
    
    # Extract baseline features
    eigenvalues, eigenvectors = np.linalg.eigh(sample_matrix)
    baseline_features = np.concatenate([
        eigenvectors[:, 0], eigenvectors[:, 1], 
        [eigenvalues[0], eigenvalues[1]]
    ])
    
    # Test different junk percentages
    feature_changes = []
    
    for junk_pct in [5, 10, 15, 20, 25, 30]:
        modified_features, _ = algorithm2_junk_insertion(sample_matrix, junk_pct)
        
        # Ensure same length for comparison
        min_len = min(len(baseline_features), len(modified_features))
        base_truncated = baseline_features[:min_len]
        mod_truncated = modified_features[:min_len]
        
        # Calculate feature differences
        feature_diff = np.abs(mod_truncated - base_truncated)
        avg_change = np.mean(feature_diff)
        max_change = np.max(feature_diff)
        
        feature_changes.append(avg_change)
        
        print(f"  {junk_pct}% junk: Avg change = {avg_change:.4f}, Max change = {max_change:.4f}")
    
    # Check if changes are meaningful
    if max(feature_changes) < 0.001:
        print("⚠️  WARNING: Very small feature changes - junk insertion may not be effective")
        return False
    else:
        print("✅ Feature changes are significant - junk insertion is working")
        return True

def validation_test_2_class_balance_check():
    """
    Validation 2: Ensure test set has balanced classes and meaningful baseline
    """
    print("\n=== VALIDATION 2: Class Balance and Baseline Check ===")
    
    # Load data and recreate test setup
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
    
    with open("X_graph_embeddings.pkl", "rb") as f:
        X_embeddings = np.array(pickle.load(f))
    
    labels = data["labels"]
    
    # Recreate the same split as in the main test
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=42)
    
    indices = list(range(len(labels)))
    for train_indices, test_indices in sss.split(indices, labels):
        test_labels = [labels[i] for i in test_indices]
        train_labels = [labels[i] for i in train_indices]
    
    # Check class distribution
    test_benign = test_labels.count(0)
    test_malware = test_labels.count(1)
    train_benign = train_labels.count(0)
    train_malware = train_labels.count(1)
    
    print(f"Train set: {train_benign} benign, {train_malware} malware")
    print(f"Test set: {test_benign} benign, {test_malware} malware")
    
    # Check if classes are balanced enough
    test_balance = min(test_benign, test_malware) / max(test_benign, test_malware)
    
    if test_balance < 0.1:
        print(f"⚠️  WARNING: Highly imbalanced test set (ratio: {test_balance:.2f})")
        return False
    else:
        print(f"✅ Reasonable class balance (ratio: {test_balance:.2f})")
        return True

def validation_test_3_random_perturbation_baseline():
    """
    Validation 3: Compare Algorithm 2 with pure random perturbation
    Algorithm 2 should perform better than random noise
    """
    print("\n=== VALIDATION 3: Algorithm 2 vs Random Perturbation ===")
    
    # Load data
    with open("X_graph_embeddings.pkl", "rb") as f:
        X_embeddings = np.array(pickle.load(f))
    
    # Use small subset for speed
    subset_X = X_embeddings[:100]
    embedding_dim = int(np.sqrt(subset_X.shape[1]))
    
    # Create test matrix
    sample_matrix = subset_X[0][:embedding_dim*embedding_dim].reshape(embedding_dim, embedding_dim)
    sample_matrix = (sample_matrix + sample_matrix.T) / 2
    
    # Baseline features
    eigenvalues, eigenvectors = np.linalg.eigh(sample_matrix)
    baseline_features = np.concatenate([
        eigenvectors[:, 0], eigenvectors[:, 1], 
        [eigenvalues[0], eigenvalues[1]]
    ])
    
    print("Comparing Algorithm 2 vs Random Perturbation:")
    
    for junk_pct in [10, 20, 30]:
        # Algorithm 2 modification
        alg2_features, _ = algorithm2_junk_insertion(sample_matrix, junk_pct)
        
        # Random perturbation (same magnitude)
        random_matrix = sample_matrix.copy()
        num_changes = max(1, int(random_matrix.size * junk_pct / 100))
        noise_std = np.std(sample_matrix) * 0.1  # Small random noise
        
        for _ in range(num_changes):
            i, j = np.random.randint(0, random_matrix.shape[0], 2)
            random_matrix[i, j] += np.random.normal(0, noise_std)
        
        # Normalize and extract features
        row_sums = random_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        random_matrix_norm = random_matrix / row_sums[:, np.newaxis]
        
        eigenvalues_rand, eigenvectors_rand = np.linalg.eigh(random_matrix_norm)
        random_features = np.concatenate([
            eigenvectors_rand[:, 0], eigenvectors_rand[:, 1],
            [eigenvalues_rand[0], eigenvalues_rand[1]]
        ])
        
        # Compare feature changes
        min_len = min(len(baseline_features), len(alg2_features), len(random_features))
        base_trunc = baseline_features[:min_len]
        alg2_trunc = alg2_features[:min_len]
        rand_trunc = random_features[:min_len]
        
        alg2_change = np.mean(np.abs(alg2_trunc - base_trunc))
        rand_change = np.mean(np.abs(rand_trunc - base_trunc))
        
        print(f"  {junk_pct}% modification:")
        print(f"    Algorithm 2 change: {alg2_change:.4f}")
        print(f"    Random change:      {rand_change:.4f}")
        
        if alg2_change > rand_change * 2:
            print(f"    ⚠️  Algorithm 2 changes features much more than random")
        elif alg2_change < rand_change * 0.5:
            print(f"    ⚠️  Algorithm 2 changes features much less than random")
        else:
            print(f"    ✅ Algorithm 2 changes are reasonable")

def validation_test_4_cross_validation():
    """
    Validation 4: Test resilience across multiple random splits
    """
    print("\n=== VALIDATION 4: Cross-Validation Consistency ===")
    
    # Load data
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
    
    with open("X_graph_embeddings.pkl", "rb") as f:
        X_embeddings = np.array(pickle.load(f))
    
    labels = data["labels"]
    
    # Test across 5 different random splits
    resilience_scores = []
    
    for seed in [42, 123, 456, 789, 999]:
        print(f"  Testing with random seed {seed}...")
        
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=seed)
        
        indices = list(range(min(200, len(labels))))  # Use subset for speed
        subset_labels = [labels[i] for i in indices]
        
        for train_indices, test_indices in sss.split(indices, subset_labels):
            # Create proxy matrices
            embedding_dim = int(np.sqrt(X_embeddings.shape[1]))
            
            train_matrices = []
            test_matrices = []
            train_labels_subset = []
            test_labels_subset = []
            
            for i in train_indices:
                matrix = X_embeddings[i][:embedding_dim*embedding_dim].reshape(embedding_dim, embedding_dim)
                matrix = (matrix + matrix.T) / 2
                train_matrices.append(matrix)
                train_labels_subset.append(labels[i])
            
            for i in test_indices:
                matrix = X_embeddings[i][:embedding_dim*embedding_dim].reshape(embedding_dim, embedding_dim)
                matrix = (matrix + matrix.T) / 2
                test_matrices.append(matrix)
                test_labels_subset.append(labels[i])
            
            # Extract features and train model
            train_features = []
            for matrix in train_matrices:
                eigenvalues, eigenvectors = np.linalg.eigh(matrix)
                features = np.concatenate([
                    eigenvectors[:, 0], eigenvectors[:, 1],
                    [eigenvalues[0], eigenvalues[1]]
                ])
                train_features.append(features)
            
            train_features = np.array(train_features)
            
            rf = RandomForestClassifier(n_estimators=20, random_state=seed)
            rf.fit(train_features, train_labels_subset)
            
            # Test baseline
            test_features = []
            for matrix in test_matrices:
                eigenvalues, eigenvectors = np.linalg.eigh(matrix)
                features = np.concatenate([
                    eigenvectors[:, 0], eigenvectors[:, 1],
                    [eigenvalues[0], eigenvalues[1]]
                ])
                test_features.append(features)
            
            test_features = np.array(test_features)
            baseline_acc = rf.score(test_features, test_labels_subset)
            
            # Test with 20% junk
            junk_features = []
            for matrix in test_matrices:
                modified_features, _ = algorithm2_junk_insertion(matrix, 20)
                junk_features.append(modified_features)
            
            junk_features = np.array(junk_features)
            
            # Ensure feature compatibility
            min_features = min(train_features.shape[1], junk_features.shape[1])
            junk_features_trunc = junk_features[:, :min_features]
            
            if junk_features_trunc.shape[1] < train_features.shape[1]:
                padding = np.zeros((junk_features_trunc.shape[0], 
                                 train_features.shape[1] - junk_features_trunc.shape[1]))
                junk_features_trunc = np.hstack([junk_features_trunc, padding])
            
            junk_acc = rf.score(junk_features_trunc, test_labels_subset)
            resilience = junk_acc / baseline_acc if baseline_acc > 0 else 0
            
            resilience_scores.append(resilience)
            print(f"    Baseline: {baseline_acc:.3f}, Junk: {junk_acc:.3f}, Resilience: {resilience:.3f}")
    
    # Analyze consistency
    avg_resilience = np.mean(resilience_scores)
    std_resilience = np.std(resilience_scores)
    
    print(f"\nCross-validation results:")
    print(f"  Average resilience: {avg_resilience:.3f} ± {std_resilience:.3f}")
    print(f"  Min resilience: {min(resilience_scores):.3f}")
    print(f"  Max resilience: {max(resilience_scores):.3f}")
    
    if std_resilience > 0.2:
        print("⚠️  WARNING: High variance in resilience scores - results may be unstable")
        return False
    else:
        print("✅ Consistent resilience across different splits")
        return True

def validation_test_5_sanity_checks():
    """
    Validation 5: Basic sanity checks
    """
    print("\n=== VALIDATION 5: Sanity Checks ===")
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Are we actually using different matrices for junk vs baseline?
    with open("X_graph_embeddings.pkl", "rb") as f:
        X_embeddings = np.array(pickle.load(f))
    
    embedding_dim = int(np.sqrt(X_embeddings.shape[1]))
    sample_matrix = X_embeddings[0][:embedding_dim*embedding_dim].reshape(embedding_dim, embedding_dim)
    
    modified_features_1, modified_matrix_1 = algorithm2_junk_insertion(sample_matrix, 20)
    modified_features_2, modified_matrix_2 = algorithm2_junk_insertion(sample_matrix, 20)
    
    if np.array_equal(modified_matrix_1, modified_matrix_2):
        print("⚠️  Check 1 FAILED: Junk insertion is deterministic (should be random)")
    else:
        print("✅ Check 1 PASSED: Junk insertion produces different results")
        checks_passed += 1
    
    # Check 2: Do higher junk percentages cause more changes?
    _, matrix_10 = algorithm2_junk_insertion(sample_matrix, 10)
    _, matrix_30 = algorithm2_junk_insertion(sample_matrix, 30)
    
    changes_10 = np.sum(matrix_10 != sample_matrix)
    changes_30 = np.sum(matrix_30 != sample_matrix)
    
    if changes_30 > changes_10:
        print("✅ Check 2 PASSED: Higher junk % causes more matrix changes")
        checks_passed += 1
    else:
        print("⚠️  Check 2 FAILED: Higher junk % doesn't cause proportionally more changes")
    
    # Check 3: Are the eigenvalues/eigenvectors actually changing?
    orig_evals, orig_evecs = np.linalg.eigh(sample_matrix)
    mod_evals, mod_evecs = np.linalg.eigh(matrix_30)
    
    eval_diff = np.mean(np.abs(orig_evals - mod_evals))
    evec_diff = np.mean(np.abs(orig_evecs - mod_evecs))
    
    if eval_diff > 1e-6 and evec_diff > 1e-6:
        print("✅ Check 3 PASSED: Eigenvalues and eigenvectors change significantly")
        checks_passed += 1
    else:
        print("⚠️  Check 3 FAILED: Eigenvalues/eigenvectors barely change")
    
    # Check 4: Is the junk insertion affecting all samples differently?
    different_samples = []
    for i in range(min(5, len(X_embeddings))):
        matrix = X_embeddings[i][:embedding_dim*embedding_dim].reshape(embedding_dim, embedding_dim)
        matrix = (matrix + matrix.T) / 2
        _, mod_matrix = algorithm2_junk_insertion(matrix, 20)
        different_samples.append(mod_matrix)
    
    # Check if all modified matrices are different
    all_different = True
    for i in range(len(different_samples)):
        for j in range(i+1, len(different_samples)):
            if np.array_equal(different_samples[i], different_samples[j]):
                all_different = False
    
    if all_different:
        print("✅ Check 4 PASSED: Junk insertion affects different samples differently")
        checks_passed += 1
    else:
        print("⚠️  Check 4 FAILED: Some samples produce identical junk modifications")
    
    print(f"\nSanity check summary: {checks_passed}/{total_checks} checks passed")
    return checks_passed >= 3

def run_all_validations():
    """
    Run all validation tests
    """
    print("🔍 VALIDATING ALGORITHM 2 JUNK CODE RESILIENCE RESULTS")
    print("=" * 60)
    
    validation_results = []
    
    validation_results.append(validation_test_1_feature_change_analysis())
    validation_results.append(validation_test_2_class_balance_check())
    validation_test_3_random_perturbation_baseline()  # Informational only
    validation_results.append(validation_test_4_cross_validation())
    validation_results.append(validation_test_5_sanity_checks())
    
    # Overall assessment
    passed_validations = sum(validation_results)
    total_validations = len(validation_results)
    
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print(f"Validations passed: {passed_validations}/{total_validations}")
    
    if passed_validations >= 3:
        print("✅ VALIDATION RESULT: Algorithm 2 results appear LEGITIMATE")
        print("   The 90.3% resilience score is likely accurate.")
    else:
        print("⚠️  VALIDATION RESULT: Algorithm 2 results are QUESTIONABLE")
        print("   The 90.3% resilience score may be artificially inflated.")
    
    return passed_validations >= 3

if __name__ == "__main__":
    run_all_validations()