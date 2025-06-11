'''
Step 1: Prepare Opcode N-Gram Matrix
    Read each .opcode file as a string of space-separated opcodes
    convert the n-gram frequency vectors (1-gram & 2-gram) 

Step 2 & 3: Implement CIG Formula | Code to Compute CIG for All Features
Step 4: Select Top 82 Features
Results:
    X_selected: matrix with top 82 features
    selected_feature_names: opcode n-grams that matter most for distinugishing malware
'''
import numpy as np
import pandas as pd
import os

from sklearn.feature_extration.text import CountVectorizer

# 'samples' --> list of strings (each sample's opode sequence)
# 'labels' --> list of 0 (benign) or 1 (malware) 

samples = []
labels = []

# read benign opcodes
for filename in os.listdir('./Benign/all_goodware/combined1'):
    if filename.endswith('.opcode'):
        with open(f'./Benign/all_goodware/combined1/{filename}') as f:
            opcode_seq = f.read().replace('\n', ' ')
            samples.append(opcode_seq)
            labels.append(0) # benign = 0

# read malware opcodes
for filename in os.listdir('./Malware(Disassembled)'):
    if filename.endswith('.opcode'):
        with open(f'./Malware(Disassembled)/{filename}') as f:
            opcode_seq = f.read().replace('\n', ' ')
            samples.append(opcode_seq)
            labels.append(1) # malware = 1

vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b')
X = vectorizer.fit_transform(samples) # shape: (n_samples, n_features)
feature_names = vectorizer.get_feature_names_out()

# step 3
def compute_cig(X, y): 
    n_samples, n_features = X.shape
    y = np.array(y)
    cig_scores = []

    # Class priors
    p_benign = np.mean(y == 0)
    p_mawlare = np.mean(y == 1)

    for j in range(n_features):
        v = X[:, j].toarray().flattern()

        # For class 0 (benign)
        f1_c0 = np.logical_and(v == 1, y == 0).sum()
        f1 = (v == 1).sum()
        p_v1_c0 = f1_c0 / n_samples
        p_v1 = f1 / n_samples

        f0_c1 = np.logical_and(v == 0, y == 1).sum()
        f0 = (v == 0).sum()
        p_v0_c1 = f0_c1 / n_samples
        p_v0 = f0 / n_samples

        term1 = p_v1_c0 * np.log((p_v1_c0 + 1e-9) / (p_v1 * p_benign + 1e-9))
        term2 = p_v0_c1 * np.log((p_v0_c1 + 1e-9) / (p_v0 * (1 - p_benign) + 1e-9))
        cig_benign = term1 + term2

        # For class 1 (malware)
        f1_c1 = np.logical_and(v == 1, y == 1).sum()
        p_v1_c1 = f1_c1 / n_samples

        f0_c0 = np.logical_and(v == 0, y == 0).sum()
        p_v0_c0 = f0_c0 / n_samples

        term1 = p_v1_c1 * np.log((p_v1_c1 + 1e-9) / (p_v1 * p_malware + 1e-9))
        term2 = p_v0_c0 * np.log((p_v0_c0 + 1e-9) / (p_v0 * (1 - p_malware) + 1e-9))
        cig_malware = term1 + term2

        cig_scores.append((j, cig_benign, cig_malware))

    return cig_scores

# step 4
cig_scores = compute_cig(X, labels)

# combine both scores
total_cig = [(idx, b + m) for idx, b, m in cig_scores]

# Sort and select top 82
top_indices = sorted(total_cig, key=lambda x: x[1], reverse=True)[:82]
selected_feature_indices = [idx for idx, _ in top_indices]

X_selected = X[:, selected_feature_indices]
selected_feature_names = [feature_names[i] for i in selected_feature_indices]