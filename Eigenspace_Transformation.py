'''
1) compute top k eigenvalues and eigenvectors
2) Flatten top-k eigenvectors into a single vector
3) Vector becomes the feature representation of that sample for training a classifier

'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

from scipy.linalg import eigh # for symmetric matrices
from sklearn.decomposition import PCA
from alive_progress import alive_bar

# Create results directory
RESULTS_DIR = "Eigenspace Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Eigenspace results will be saved to: {RESULTS_DIR}/")

# eigenspace transformation function
def eigenspace_embedding(adj_matrices, k=2):  # Changed from k=12 to k=2
    embedded_vectors = []

    print(f"\n==Computing Eigenspace Embeddings (k={k})==")
    print(f"Following paper methodology: Using top {k} eigenvectors")
    
    with alive_bar(len(adj_matrices), title="Embedding graphs") as bar:
        for i, A in enumerate(adj_matrices):
            # ensure A is symmetric for stable eigendecomposition
            A_sym = (A + A.T) / 2

            # compute top-k eigenvalues & eigenvectors
            eigenvalues, eigenvectors = eigh(A_sym)

            # sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            top_k_vectors = eigenvectors[:, idx[:k]]

            # === Print details for first sample only ===
            if i == 0:
                print(f"\nFirst sample analysis:")
                print(f"   Matrix shape: {A_sym.shape}")
                print(f"   Matrix sum: {np.sum(A_sym):.3f}")
                
                top_k_values = eigenvalues[idx[:k]]
                
                print(f"   Top {k} eigenvalues: {top_k_values}")
                print(f"   Top {k} eigenvectors shape: {top_k_vectors.shape}")
                print(f"   Feature vector dimension: {top_k_vectors.size} (matrix_size × k = {A.shape[0]} × {k})")
                
                # Show eigenvalue dominance
                total_eigenvalue_sum = np.sum(np.abs(eigenvalues))
                top_k_sum = np.sum(np.abs(top_k_values))
                dominance_ratio = top_k_sum / total_eigenvalue_sum * 100
                print(f"   Top {k} eigenvalues capture {dominance_ratio:.1f}% of spectral energy")

            # flatten to 1D vector
            embedding = top_k_vectors.flatten()
            embedded_vectors.append(embedding)
            bar()
        
    return np.array(embedded_vectors)

def visualize_embeddings_with_labels(X, labels):
    print("\n==Visualizing Eigenspace Embeddings with Class Labels==")
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(12, 5))
    
    # Plot 1: All samples (original)
    plt.subplot(1, 2, 1)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=15, alpha=0.7, color='blue')
    plt.title("All Samples")
    plt.xlabel("Structural Variation in Opcode Flow")
    plt.ylabel("Secondary Behavioral Pattern")
    plt.grid(True)
    
    # Plot 2: Colored by class
    plt.subplot(1, 2, 2)
    benign_mask = np.array(labels) == 0
    malware_mask = np.array(labels) == 1
    
    plt.scatter(X_reduced[benign_mask, 0], X_reduced[benign_mask, 1], 
               s=15, alpha=0.7, color='green', label=f'Benign ({benign_mask.sum()})')
    plt.scatter(X_reduced[malware_mask, 0], X_reduced[malware_mask, 1], 
               s=15, alpha=0.7, color='red', label=f'Malware ({malware_mask.sum()})')
    
    plt.title("Malware vs Benign Classification")
    plt.xlabel("Structural Variation in Opcode Flow")
    plt.ylabel("Secondary Behavioral Pattern")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "eigenspace_embeddings_labeled.png"), dpi=300)
    print("Saved labeled visualization to 'Eigenspace Results/eigenspace_embeddings_labeled.png'\n")
    
    # Print separation statistics
    print("=== Cluster Analysis ===")
    print(f"Total samples: {len(labels)}")
    print(f"Benign: {benign_mask.sum()} ({benign_mask.sum()/len(labels)*100:.1f}%)")
    print(f"Malware: {malware_mask.sum()} ({malware_mask.sum()/len(labels)*100:.1f}%)")
    
    # Calculate basic separation metrics
    benign_centroid = X_reduced[benign_mask].mean(axis=0)
    malware_centroid = X_reduced[malware_mask].mean(axis=0)
    separation_distance = np.linalg.norm(benign_centroid - malware_centroid)
    
    print(f"Benign centroid: ({benign_centroid[0]:.3f}, {benign_centroid[1]:.3f})")
    print(f"Malware centroid: ({malware_centroid[0]:.3f}, {malware_centroid[1]:.3f})")
    print(f"Centroid separation distance: {separation_distance:.3f}")

def visualize_embeddings(X):
    print("\n==Visualizing Eigenspace Embeddings==")
    pca = PCA(n_components=2)  # or use TSNE for nonlinear projection
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=15, alpha=0.7)
    plt.title("Eigenspace Embeddings (PCA Projection)")
    plt.xlabel("Structural Variation in Opcode Flow")
    plt.ylabel("Secondary Behavioral Pattern")
    plt.grid(True)
    plt.tight_layout()

    # Save instead of showing
    plt.savefig(os.path.join(RESULTS_DIR, "eigenspace_embeddings.png"), dpi=300)
    print("Saved PCA visualization to 'Eigenspace Results/eigenspace_embeddings.png'\n")

def main():
    # load data --> get adj_matrices and labels
    with open("opcode_graphs_optimized.pkl", "rb") as f:
        adj_matrices = pickle.load(f)
    
    # load labels for visualization
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
        labels = data["labels"]

    # compute embeddings with k=2 as per paper
    X_graph = eigenspace_embedding(adj_matrices, k=2)  # Changed to k=2
    
    # Calculate actual feature dimensions
    matrix_size = adj_matrices[0].shape[0]
    expected_dims = matrix_size * 2
    actual_dims = X_graph.shape[1]
    
    print(f"\nEigenspace embeddings computed:")
    print(f"   Shape: {X_graph.shape}")
    print(f"   Expected dimensions: {expected_dims} (matrix_size {matrix_size} × k=2)")
    print(f"   Actual dimensions: {actual_dims}")
    print(f"   ✅ Dimensions match: {expected_dims == actual_dims}")
    
    # save embeddings
    with open("X_graph_embeddings.pkl", "wb") as f:
        pickle.dump(X_graph, f)

    print(f"\n✅ Saved eigenspace embeddings to 'X_graph_embeddings.pkl'")
    print(f"   Each sample represented by {actual_dims}-dimensional vector")
    print(f"   Following paper methodology: k=2 eigenvectors")

    # Create visualizations
    visualize_embeddings(X_graph)
    visualize_embeddings_with_labels(X_graph, labels)

if __name__ == "__main__":
    main()