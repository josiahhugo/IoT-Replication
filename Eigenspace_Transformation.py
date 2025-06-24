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

from scipy.linalg import eigh # for symmetric matrices
from sklearn.decomposition import PCA
from alive_progress import alive_bar

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
    plt.savefig("eigenspace_embeddings.png", dpi=300)
    print("Saved PCA visualization to 'eigenspace_embeddings.png'\n")

def main():
    # load data --> get adj_matrices
    with open("opcode_graphs.pkl", "rb") as f:
        adj_matrices = pickle.load(f)

    # compute embeddings
    X_graph = eigenspace_embedding(adj_matrices, k=12)
    print(f"Eigenspace embeddings shape: {X_graph.shape}") # each sample is now a 984-dimensional vector ready to train a classifier

    # save embeddings
    with open("X_graph_embeddings.pkl", "wb") as f:
        pickle.dump(X_graph, f)

    print("\nSaved eigenspace embeddings to 'X_graph_embeddings.pkl'")

    visualize_embeddings(X_graph)

if __name__ == "__main__":
    main()