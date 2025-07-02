'''
Proper Implementation of Algorithm 2: Junk Code Insertion Procedure
Following the exact algorithm from the paper
'''
import pickle
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Eigenspace_Transformation import eigenspace_embedding
import json

def algorithm2_junk_insertion(adjacency_matrix, junk_percentage_k):
    """
    Implementation of Algorithm 2 from the paper
    
    Input: Trained Classifier D, Test Samples S, Junk Code Percentage k
    Output: Predicted Class for Test Samples P
    """
    # Step 1: P = {} (initialize predictions)
    
    # Step 2-4: For each sample, compute CFG and select k% of W's indices
    W = adjacency_matrix.copy()  # W is the adjacency matrix
    
    # Step 4: R = select k% of W's indices randomly (Allow duplicate indices)
    total_elements = W.size
    num_to_select = max(1, int(total_elements * junk_percentage_k / 100))
    
    # Get random indices (flattened matrix)
    selected_indices = np.random.randint(0, total_elements, size=num_to_select)
    
    # Step 5-7: For each selected index, increment weight
    for idx in selected_indices:
        # Convert flat index to 2D coordinates
        row, col = np.unravel_index(idx, W.shape)
        W[row, col] += 1  # W_index = W_index + 1
    
    # Step 8: Normalize W
    W_normalized = normalize_adjacency_matrix(W)
    
    # Step 9-11: Extract eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(W_normalized)
    
    # Take first and second eigenvectors (as mentioned in algorithm)
    e1 = eigenvectors[:, 0] if eigenvectors.shape[1] > 0 else np.zeros(W.shape[0])
    e2 = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(W.shape[0])
    
    # Take first and second eigenvalues
    l1 = eigenvalues[0] if len(eigenvalues) > 0 else 0
    l2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
    
    # Step 11: P = P ∪ D(e1, e2, l1, l2) - combine features for classification
    modified_features = np.concatenate([e1, e2, [l1, l2]])
    
    return modified_features, W_normalized

def normalize_adjacency_matrix(matrix):
    """
    Normalize adjacency matrix (various methods possible)
    """
    # Method 1: Row normalization
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = matrix / row_sums[:, np.newaxis]
    
    return normalized

def test_paper_algorithm2():
    """
    Test the paper's exact Algorithm 2 implementation
    """
    print("=== Testing Paper's Algorithm 2: Junk Code Insertion ===")
    print("Following the exact algorithm from the paper")
    
    # Load data
    print("Loading data...")
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Load adjacency matrices if available, otherwise use embeddings
    try:
        with open("adjacency_matrices.pkl", "rb") as f:
            adjacency_matrices = pickle.load(f)
    except FileNotFoundError:
        print("Adjacency matrices not found, using embeddings as proxy...")
        with open("X_graph_embeddings.pkl", "rb") as f:
            X_embeddings = np.array(pickle.load(f))
        
        # Create proxy adjacency matrices from embeddings
        # This is a simplification - in practice, would need actual graph matrices
        adjacency_matrices = []
        embedding_dim = int(np.sqrt(X_embeddings.shape[1]))  # Assume square matrices
        
        for embedding in X_embeddings:
            # Reshape embedding back to matrix form (approximation)
            if len(embedding) >= embedding_dim * embedding_dim:
                matrix = embedding[:embedding_dim*embedding_dim].reshape(embedding_dim, embedding_dim)
                # Make symmetric
                matrix = (matrix + matrix.T) / 2
                adjacency_matrices.append(matrix)
            else:
                # Fallback: create identity matrix
                adjacency_matrices.append(np.eye(embedding_dim))
    
    samples = data["samples"]
    labels = data["labels"]
    
    print(f"Dataset: {len(samples)} samples")
    print(f"Adjacency matrices: {len(adjacency_matrices)} matrices")
    
    # Create balanced subset for testing
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=42)
    
    indices = list(range(min(len(samples), len(adjacency_matrices), len(labels))))
    subset_labels = [labels[i] for i in indices]
    
    for train_indices, test_indices in sss.split(indices, subset_labels):
        test_matrices = [adjacency_matrices[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        train_matrices = [adjacency_matrices[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
    
    print(f"Train: {len(train_matrices)}, Test: {len(test_matrices)}")
    
    # Extract features from original matrices for training
    print("Extracting baseline features...")
    train_features = []
    for matrix in train_matrices:
        try:
            # Extract eigenspace features as in the algorithm
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            e1 = eigenvectors[:, 0] if eigenvectors.shape[1] > 0 else np.zeros(matrix.shape[0])
            e2 = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(matrix.shape[0])
            l1 = eigenvalues[0] if len(eigenvalues) > 0 else 0
            l2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
            
            features = np.concatenate([e1, e2, [l1, l2]])
            train_features.append(features)
        except Exception as e:
            print(f"Error processing matrix: {e}")
            # Fallback to zero features
            train_features.append(np.zeros(matrix.shape[0] * 2 + 2))
    
    train_features = np.array(train_features)
    
    # Train classifier D
    print("Training classifier...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(train_features, train_labels)
    
    # Test baseline (without junk insertion)
    test_features_baseline = []
    for matrix in test_matrices:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            e1 = eigenvectors[:, 0] if eigenvectors.shape[1] > 0 else np.zeros(matrix.shape[0])
            e2 = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(matrix.shape[0])
            l1 = eigenvalues[0] if len(eigenvalues) > 0 else 0
            l2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
            
            features = np.concatenate([e1, e2, [l1, l2]])
            test_features_baseline.append(features)
        except:
            test_features_baseline.append(np.zeros(matrix.shape[0] * 2 + 2))
    
    test_features_baseline = np.array(test_features_baseline)
    baseline_accuracy = rf.score(test_features_baseline, test_labels)
    
    print(f"Baseline accuracy: {baseline_accuracy:.3f}")
    
    # Test Algorithm 2 with different junk percentages
    print("\n=== Testing Algorithm 2 Junk Insertion ===")
    
    results = {"baseline_accuracy": baseline_accuracy}
    junk_percentages = [5, 10, 15, 20, 25, 30]
    
    for k in junk_percentages:
        print(f"\nTesting {k}% junk insertion (Algorithm 2)...")
        
        # Apply Algorithm 2 to each test sample
        modified_features = []
        
        for matrix in test_matrices:
            # Apply Algorithm 2
            modified_feature_vector, modified_matrix = algorithm2_junk_insertion(matrix, k)
            modified_features.append(modified_feature_vector)
        
        modified_features = np.array(modified_features)
        
        # Ensure feature dimensions match
        if modified_features.shape[1] != train_features.shape[1]:
            # Pad or truncate to match training features
            min_features = min(modified_features.shape[1], train_features.shape[1])
            modified_features = modified_features[:, :min_features]
            
            if modified_features.shape[1] < train_features.shape[1]:
                padding = np.zeros((modified_features.shape[0], 
                                  train_features.shape[1] - modified_features.shape[1]))
                modified_features = np.hstack([modified_features, padding])
        
        # Predict using classifier D
        try:
            predictions = rf.predict(modified_features)
            accuracy = accuracy_score(test_labels, predictions)
            resilience = accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
            
            results[f"junk_{k}pct"] = {
                "accuracy": round(accuracy, 4),
                "resilience": round(resilience, 4),
                "drop_pct": round((baseline_accuracy - accuracy) * 100, 2)
            }
            
            print(f"  {k}% junk: {accuracy:.3f} accuracy ({resilience:.3f} resilience)")
            
        except Exception as e:
            print(f"  Error with {k}% junk: {e}")
            results[f"junk_{k}pct"] = {"error": str(e)}
    
    # Summary
    print("\n=== ALGORITHM 2 RESULTS ===")
    print(f"Baseline accuracy: {baseline_accuracy:.3f}")
    print("Junk%  Accuracy  Resilience  Drop%")
    print("-" * 35)
    
    valid_results = []
    for k in junk_percentages:
        key = f"junk_{k}pct"
        if key in results and "error" not in results[key]:
            acc = results[key]["accuracy"]
            res = results[key]["resilience"]
            drop = results[key]["drop_pct"]
            valid_results.append(res)
            print(f"{k:3d}%   {acc:.3f}    {res:.3f}     {drop:+.1f}%")
    
    if valid_results:
        avg_resilience = np.mean(valid_results)
        print(f"\nAverage resilience: {avg_resilience:.3f}")
        
        if avg_resilience > 0.85:
            assessment = "EXCELLENT - Algorithm 2 shows high resilience to junk insertion"
        elif avg_resilience > 0.70:
            assessment = "GOOD - Algorithm 2 shows moderate resilience"
        elif avg_resilience > 0.50:
            assessment = "MODERATE - Some resilience to junk insertion"
        else:
            assessment = "POOR - Low resilience to junk insertion"
        
        results["assessment"] = assessment
        results["average_resilience"] = avg_resilience
        
        print(f"Assessment: {assessment}")
    
    # Save results
    with open("algorithm2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: algorithm2_results.json")
    return results

if __name__ == "__main__":
    test_paper_algorithm2()