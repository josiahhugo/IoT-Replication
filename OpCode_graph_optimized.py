# **OPTIMIZED GRAPH CONSTRUCTION**

import pickle
import numpy as np
import math
from collections import defaultdict

from alive_progress import alive_bar # for visual feedback

# Load data --> samples and selected_feature_names are available in graph construction
with open("improved_cig_output.pkl", "rb") as f:
    data = pickle.load(f)

samples = data["samples"] # list of opcode sequences (1 per sample)
selected_feature_names = data["selected_feature_names"] # top 82 n-gram features

# 1) Build index mapping from features to indices
feature_to_index = {feat: i for i, feat in enumerate(selected_feature_names)} 

# 3) Create 82x82 adjacency matrix for a sample using paper's methodology - OPTIMIZED
def build_graph_matrix_optimized(opcode_seq, feature_to_index):
    n = len(feature_to_index)
    adj_matrix = np.zeros((n, n)) # initialize square matrix

    feature_set = set(feature_to_index.keys())
    tokens = opcode_seq.split()
    
    # OPTIMIZATION 1: Early exit if sequence is too long
    if len(tokens) > 100000:
        print(f"Warning: Sequence too long ({len(tokens)} tokens), truncating to 100k")
        tokens = tokens[:100000]
    
    # OPTIMIZATION 2: Batch feature position tracking
    feature_positions = defaultdict(list)
    
    # Track 1-gram positions efficiently
    for i, token in enumerate(tokens):
        if token in feature_set:
            feature_positions[token].append(i)
    
    # Track 2-gram positions efficiently
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in feature_set:
            feature_positions[bigram].append(i)
    
    # OPTIMIZATION 3: Only process features that actually appear
    present_features = list(feature_positions.keys())
    if len(present_features) < 2:
        # Not enough features to create meaningful edges
        return adj_matrix / (adj_matrix.sum(axis=1, keepdims=True) + 1e-9)
    
    # OPTIMIZATION 4: Vectorized distance calculations with limits
    alpha = 1.0
    max_distance = 50  # Paper focuses on local relationships
    
    for i, feat_i in enumerate(present_features):
        for j, feat_j in enumerate(present_features):
            if i >= j:  # Skip diagonal and lower triangle (we'll make symmetric later)
                continue
                
            positions_i = np.array(feature_positions[feat_i])
            positions_j = np.array(feature_positions[feat_j])
            
            # OPTIMIZATION 5: Vectorized distance calculation
            # Create all pairwise distances using broadcasting
            distances = np.abs(positions_i[:, np.newaxis] - positions_j[np.newaxis, :])
            
            # Filter out distances > max_distance
            valid_distances = distances[distances <= max_distance]
            
            if len(valid_distances) > 0:
                # Vectorized weight calculation
                weights = 2.0 / (1.0 + alpha * np.exp(valid_distances))
                total_weight = np.sum(weights)
                
                # Set symmetric weights
                idx_i = feature_to_index[feat_i]
                idx_j = feature_to_index[feat_j]
                adj_matrix[idx_i][idx_j] = total_weight
                adj_matrix[idx_j][idx_i] = total_weight  # Make symmetric
    
    # Apply row normalization to create transition probabilities
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for isolated nodes
    adj_matrix = adj_matrix / row_sums

    return adj_matrix

# Alternative: Even faster approximate version
def build_graph_matrix_fast(opcode_seq, feature_to_index):
    n = len(feature_to_index)
    adj_matrix = np.zeros((n, n))

    feature_set = set(feature_to_index.keys())
    tokens = opcode_seq.split()
    
    # SUPER OPTIMIZATION: Sample tokens if sequence is very long
    if len(tokens) > 50000:
        # Sample every nth token to maintain sequence structure
        step = len(tokens) // 50000 + 1
        tokens = tokens[::step]
    
    feature_positions = defaultdict(list)
    
    # Track positions
    for i, token in enumerate(tokens):
        if token in feature_set:
            feature_positions[token].append(i)
    
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in feature_set:
            feature_positions[bigram].append(i)
    
    present_features = list(feature_positions.keys())
    
    # FAST APPROXIMATION: Only consider nearby features (sliding window)
    alpha = 1.0
    window_size = 1000  # Only look within 1000 token windows
    
    for feat_i in present_features:
        for feat_j in present_features:
            if feat_i == feat_j:
                continue
                
            positions_i = feature_positions[feat_i]
            positions_j = feature_positions[feat_j]
            
            total_weight = 0.0
            
            # For each position of feat_i, only check feat_j positions within window
            for pos_i in positions_i:
                nearby_positions_j = [pos_j for pos_j in positions_j 
                                    if abs(pos_j - pos_i) <= window_size]
                
                for pos_j in nearby_positions_j:
                    distance = abs(pos_j - pos_i)
                    if distance <= 50:  # Keep the distance limit
                        weight = 2.0 / (1.0 + alpha * math.exp(distance))
                        total_weight += weight
            
            if total_weight > 0:
                idx_i = feature_to_index[feat_i]
                idx_j = feature_to_index[feat_j]
                adj_matrix[idx_i][idx_j] = total_weight
    
    # Row normalization
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_matrix = adj_matrix / row_sums

    return adj_matrix

# main to generate all sample graphs
def main():
    print("==Building OpCode transition graphs with optimized algorithm==")
    
    # Let user choose optimization level
    print("Choose optimization level:")
    print("1. Optimized (recommended) - ~10x faster")
    print("2. Fast approximation - ~50x faster, slight quality loss")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        build_func = build_graph_matrix_fast
        print("Using fast approximation algorithm...")
    else:
        build_func = build_graph_matrix_optimized
        print("Using optimized algorithm...")
    
    adj_matrices = []

    with alive_bar(len(samples), title="Building graphs") as bar:
        for i, sample in enumerate(samples):
            matrix = build_func(sample, feature_to_index)
            adj_matrices.append(matrix)
            bar()
            
            # Progress update every 100 samples
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{len(samples)} samples")

    print(f"\nBuilt {len(adj_matrices)} opcode graphs with shape {adj_matrices[0].shape} each.")

    # save the adjacency matrices
    output_file = "opcode_graphs_optimized.pkl" if choice == "1" else "opcode_graphs_fast.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(adj_matrices, f)

    print(f"\nSaved opcode transition graphs to '{output_file}'")

if __name__ == "__main__":
    main()
