# **CONSTRUCTING THE GRAPH**

import pickle
import numpy as np
import math

from alive_progress import alive_bar # for visual feedback

# Load data --> samples and selected_feature_names are available in graph construction
with open("improved_cig_output.pkl", "rb") as f:
    data = pickle.load(f)

samples = data["samples"] # list of opcode sequences (1 per sample)
selected_feature_names = data["selected_feature_names"] # top 82 n-gram features

# 1) Build index mapping from features to indices
feature_to_index = {feat: i for i, feat in enumerate(selected_feature_names)} 

# 2) Extract valid 1-grams and 2-grams from opcode sequences
def extract_relevant_ngrams(opcode_seq, feature_set):
    tokens = opcode_seq.split() # split opcode string into individual opcodes
    ngrams = []

    # keep valid 1-grams
    ngrams.extend([t for t in tokens if t in feature_set])

    # keep valid 2-grams
    for i in range(len(tokens) -1):
        bigram = f'{tokens[i]} {tokens[i+1]}'
        if bigram in feature_set:
            ngrams.append(bigram)

    return ngrams        

# 3) Create 82x82 adjacency matrix for a sample using paper's methodology
def build_graph_matrix(opcode_seq, feature_to_index):
    n = len(feature_to_index)
    adj_matrix = np.zeros((n, n)) # initialize square matrix

    feature_set = set(feature_to_index.keys())
    tokens = opcode_seq.split()
    
    # According to the paper, we create edges between features that appear in the opcode sequence
    # The edge weight is based on their co-occurrence and distance relationship
    
    # First, find all occurrences of our selected features in the sequence
    feature_positions = {}
    
    # Track 1-gram positions
    for i, token in enumerate(tokens):
        if token in feature_set:
            if token not in feature_positions:
                feature_positions[token] = []
            feature_positions[token].append(i)
    
    # Track 2-gram positions  
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in feature_set:
            if bigram not in feature_positions:
                feature_positions[bigram] = []
            feature_positions[bigram].append(i)
    
    # Build adjacency matrix using paper's formula: E(vi,vj) = Σ 2/(1 + α * e^|distance|)
    # For adjacent opcodes (distance=1) with α=1: weight = 2/(1 + e^1) ≈ 0.462
    alpha = 1.0  # Paper uses α=1 for adjacent opcodes
    
    # Create edges between all pairs of features that appear in this sequence
    present_features = list(feature_positions.keys())
    
    for feat_i in present_features:
        for feat_j in present_features:
            if feat_i != feat_j:  # No self-loops
                total_weight = 0.0
                
                # Calculate weight based on all position pairs between these features
                for pos_i in feature_positions[feat_i]:
                    for pos_j in feature_positions[feat_j]:
                        distance = abs(pos_j - pos_i)
                        
                        # Limit distance to prevent overflow and focus on nearby relationships
                        # Paper focuses on local relationships, distant pairs contribute negligibly
                        if distance > 50:  # Skip very distant pairs
                            continue
                            
                        # Apply exponential decay formula with overflow protection
                        try:
                            weight = 2.0 / (1.0 + alpha * math.exp(distance))
                            total_weight += weight
                        except OverflowError:
                            # For very large distances, weight approaches 0
                            continue
                
                # Set edge weight in adjacency matrix
                if total_weight > 0:
                    idx_i = feature_to_index[feat_i]
                    idx_j = feature_to_index[feat_j]
                    adj_matrix[idx_i][idx_j] = total_weight
    
    # Apply row normalization to create transition probabilities
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for isolated nodes
    adj_matrix = adj_matrix / row_sums

    return adj_matrix

# main to generate all sample graphs
def main():
    print("==Building OpCode transition graphs with exponential weights==")
    adj_matrices = []

    with alive_bar(len(samples), title="Building graphs") as bar:
        for sample in samples:
            matrix = build_graph_matrix(sample, feature_to_index)
            adj_matrices.append(matrix)
            bar()

    print(f"\nBuilt {len(adj_matrices)} opcode graphs with shape {adj_matrices[0].shape} each.")

    # save the adjacency matrices
    with open("opcode_graphs.pkl", "wb") as f:
        pickle.dump(adj_matrices, f)

    print("\nSaved opcode transition graphs to 'opcode_graphs.pkl'")

if __name__ == "__main__":
    main()
