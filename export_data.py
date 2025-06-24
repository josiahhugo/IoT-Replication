# export data from python for matlab --> generate .mat file do directly load in MATLAB
import pickle
import numpy as np
from scipy.io import savemat

# Load your data from Python
with open('X_graph_embeddings.pkl', 'rb') as f:
    X = pickle.load(f)  # shape (n_samples, n_features)

with open('cig_output.pkl', 'rb') as f:
    data = pickle.load(f)
    Y = data["labels"] # get labesl from the combined pickle


def main():
    print("\n==Exporting data for MATLAB==")
    # Optional: convert labels to string for MATLAB
    labels = np.array(['malware' if label == 1 else 'benign' for label in Y])

    # Save as .mat for MATLAB
    savemat('malware_dataset.mat', {
        'X': X,
        'Y': labels
    })
    print("Data successfully exported to 'malware_dataset.mat'")
if __name__ == "__main__":
    main()