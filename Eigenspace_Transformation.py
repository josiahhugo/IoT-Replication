'''
1) compute top k eigenvalues and eigenvectors
2) Flatten top-k eigenvectors into a single vector
3) Vector becomes the feaeture representation of that sample for training a classifier

paper uses k = 12

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
def eigenspace_embedding(adj_matrices, k=12):
    embedded_vectors = []

    print("\n==Computing Eigenspace Embeddings==")
    with alive_bar(len(adj_matrices), title="Embedding graphs") as bar:
        for i, A in enumerate(adj_matrices):
            # ensure A is symmetric for stable eigendecomposition
            A_sym = (A + A.T) / 2

            # compute top-k eigenvalues & eigenvectors
            eigenvalues, eigenvectors = eigh(A_sym)

            # sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            top_k_vectors = eigenvectors[:, idx[:k]]

            # === Print or inspect top 2 eigenvalues and eigenvectors (for first sample only) ===
            if i == 0:
                print(f"A_sym[0] (first 5 rows):\n{A_sym[:5, :5]}")
                print(f"Sum of A_sym: {np.sum(A_sym)}")
                
                top_2_values = eigenvalues[idx[:2]]
                top_2_vectors = eigenvectors[:, idx[:2]]

                print(f"\nSample eigenvalues (top 2): {top_2_values}")
                print(f"Top 2 eigenvectors (shape {top_2_vectors.shape}):\n{top_2_vectors}")

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

    # compute embeddings
    X_graph = eigenspace_embedding(adj_matrices, k=12)
    print(f"Eigenspace embeddings shape: {X_graph.shape}") # each sample is now a 984-dimensional vector ready to train a classifier

    # save embeddings
    with open("X_graph_embeddings.pkl", "wb") as f:
        pickle.dump(X_graph, f)

    print("\nSaved eigenspace embeddings to 'X_graph_embeddings.pkl'")

    # Create both visualizations
    visualize_embeddings(X_graph)
    visualize_embeddings_with_labels(X_graph, labels)

    # For class-aware visualization, assuming labels are available in the data
    # Here, we just create dummy labels for demonstration (0 for benign, 1 for malware)
    dummy_labels = np.array([0] * (len(adj_matrices) // 2) + [1] * (len(adj_matrices) - len(adj_matrices) // 2))
    visualize_embeddings_with_labels(X_graph, dummy_labels)

if __name__ == "__main__":
    main()