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
import pickle # to save the data needed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from alive_progress import alive_bar; import time

# 'samples' --> list of strings (each sample's opode sequence)
# 'labels' --> list of 0 (benign) or 1 (malware) 

samples = []
labels = []

# step 3
def compute_cig(X, y): 
    n_samples, n_features = X.shape
    y = np.array(y)
    cig_scores = []

    # Class priors
    p_benign = np.mean(y == 0)
    p_malware = np.mean(y == 1)

    with alive_bar(n_features, title='Calculating CIG') as bar:
        for j in range(n_features):
            v = X[:, j].toarray().flatten()

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

            bar() # progress bar

    return cig_scores

def main():
    # read benign opcodes
    print("==Reading benign opcode files==")
    benign_subdirs = ['combined1', 'combined2', 'combined3']
    benign_files = []

    for subdir in benign_subdirs:
        path = f'/home/josiah/research/IoTMalwareDetection-master/IoTMalwareDetection-master/Benign/all_goodware/{subdir}'
        benign_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.opcode')])

    with alive_bar(len(benign_files), title="Loading benign samples") as bar:
        for filepath in benign_files:
            with open(filepath) as f:
                opcode_seq = f.read().replace('\n', ' ')
                samples.append(opcode_seq)
                labels.append(0)  # benign = 0
            bar()

    # read malware opcodes with progress bar
    print("\n==Reading malware opcode files==")
    malware_path = '/home/josiah/research/IoTMalwareDetection-master/IoTMalwareDetection-master/Malware(Disassembeled)/'
    malware_files = [f for f in os.listdir(malware_path) if f.endswith('.opcode')]

    with alive_bar(len(malware_files), title="Loading malware samples") as bar:
        for filename in os.listdir(malware_path):
            if filename.endswith('.opcode'):
                with open(os.path.join(malware_path, filename)) as f:
                    opcode_seq = f.read().replace('\n', ' ')
                    samples.append(opcode_seq)
                    labels.append(1) # malware = 1
                bar()    

    # Show total samples
    print(f"\nTotal samples: {len(samples)} \nBenign: {labels.count(0)} \nMalware: {labels.count(1)}\n")

    # Vectorization
    print("==Vectorizing opcode sequences into n-grams (1-gram & 2-gram)==")
    with alive_bar(1, title="Vectorizing") as bar:
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', max_features=10000) # NEW: max_features--> adjust as needed
        X = vectorizer.fit_transform(samples) # shape: (n_samples, n_features)
        bar()
    
    feature_names = vectorizer.get_feature_names_out()
    print(f"Feature vector shape: {X.shape} (samples x features)\n")


    # CIG computation
    print("==Computing Class-Wise Information Gain (CIG)==")
    cig_scores = compute_cig(X, labels)

    # combine both scores
    print("\n==Selecting top 82 features based on CIG scores==")
    with alive_bar(1, title="Selecting features") as bar:
        total_cig = [(idx, b + m) for idx, b, m in cig_scores]

        # Sort and select top 82
        top_indices = sorted(total_cig, key=lambda x: x[1], reverse=True)[:82]
        selected_feature_indices = [idx for idx, _ in top_indices]

        # Reduce the feature matrix to top 82 features
        X_selected = X[:, selected_feature_indices]
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        bar()

    # Output Sumamry
    print(f"\nSelected top 82 features.")
    print(selected_feature_names)

    print(f"\nFinal selected feature matrix shape: {X_selected.shape}")

    # save data --> saves samples and selected_feature_names into a file using pickle format
    with open("cig_output.pkl", "wb") as f:
        pickle.dump({
            "samples": samples,
            "selected_feature_names": selected_feature_names,
            "labels": labels
        }, f)

    print("\nSaved CIG-selected features and samples to 'cig_output.pkl'\n")

if __name__ == "__main__":
    main()