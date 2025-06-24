# **CONSTRUCTING THE GRAPH**

import pickle
import numpy as np
import math

from alive_progress import alive_bar # for visual feedback

# Load data --> samples and selected_feature_names are available in graph construction
with open("cig_output.pkl", "rb") as f:
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

# 3) Create 82x82 adjacency matrix for a sample using exponential distance weights
def build_graph_matrix(opcode_seq, feature_to_index):
    n = len(feature_to_index)
    adj_matrix = np.zeros((n, n)) # initialize square matrix

    feature_set = set(feature_to_index.keys())
    tokens = opcode_seq.split()
    positions = {}

    # map positions of each n-gram
    for i in range(len(tokens)):
        unigram = tokens[i]
        if unigram in feature_set:
            positions.setdefault(unigram, []).append(i)
        if i < len(tokens) - 1:
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in feature_set:
                positions.setdefault(bigram, []).append(i)

    # compute weighted edges with e^{-distance}
    for src_feat in feature_to_index:
        for dst_feat in feature_to_index:
            if src_feat in positions and dst_feat in positions:
                min_distance = float('inf')
                for s in positions[src_feat]:
                    for t in positions[dst_feat]:
                        if t > s:
                            d = t - s - 1
                            if d < min_distance:
                                min_distance = d
                if min_distance != float('inf'):
                    weight = math.exp(-min_distance)
                    src_idx = feature_to_index[src_feat]
                    dst_idx = feature_to_index[dst_feat]
                    adj_matrix[src_idx][dst_idx] += weight

    # row normalize to get probability distribution
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
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
