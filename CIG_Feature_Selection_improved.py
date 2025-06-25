'''
Improved CIG Feature Selection with proper preprocessing
Based on original CIG_Feature_Selection.py but with better opcode filtering
to match research paper methodology more closely.

Step 1: Clean and Prepare Opcode N-Gram Matrix
    - Filter out hex addresses, register names, and non-instruction tokens
    - Keep only ARM instruction mnemonics
    - Convert to n-gram frequency vectors (1-gram & 2-gram)

Step 2 & 3: Implement CIG Formula | Code to Compute CIG for All Features
Step 4: Select Top 82 Features
Results:
    X_selected: matrix with top 82 features
    selected_feature_names: opcode n-grams that matter most for distinguishing malware
'''
import numpy as np
import pandas as pd
import os
import pickle
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from alive_progress import alive_bar
import time

# 'samples' --> list of strings (each sample's opcode sequence)
# 'labels' --> list of 0 (benign) or 1 (malware) 

samples = []
labels = []

def clean_opcode_sequence(opcode_sequence):
    """
    Clean opcode sequence to remove hex addresses and keep only instruction mnemonics
    """
    tokens = opcode_sequence.split()
    cleaned_tokens = []
    
    # ARM instruction patterns
    arm_instructions = {
        'mov', 'ldr', 'str', 'add', 'sub', 'cmp', 'bl', 'blx', 'push', 'pop',
        'b', 'bx', 'beq', 'bne', 'blt', 'bgt', 'ble', 'bge', 'bls', 'bhi',
        'movs', 'adds', 'subs', 'lsl', 'lsr', 'asr', 'ror', 'and', 'orr',
        'eor', 'bic', 'mvn', 'tst', 'teq', 'cmn', 'ldm', 'stm', 'swi',
        'msr', 'mrs', 'mul', 'mla', 'umull', 'umlal', 'smull', 'smlal',
        'movw', 'movt', 'ldrb', 'strb', 'ldrh', 'strh', 'ldrsb', 'ldrsh',
        'cbz', 'cbnz', 'dmb', 'dsb', 'isb', 'plt', 'base', 'undefined',
        'instruction', 'error', 'svclt', 'blls', 'submi', 'popeq', 'cmpcs',
        'andeq', 'lslne'
    }
    
    for token in tokens:
        token = token.lower().strip()
        
        # Skip empty tokens
        if not token:
            continue
            
        # Skip pure hex addresses (like 0x1234, 00000000) but keep short hex that might be opcodes
        if re.match(r'^(0x)?[0-9a-f]+$', token) and len(token) > 6:
            continue
            
        # Skip register names (r0, r1, etc.) but keep them if part of instruction
        if re.match(r'^r\d+$', token):
            continue
            
        # Skip special registers
        if token in ['sp', 'lr', 'pc', 'fp', 'ip', 'sl'] and len(token) <= 3:
            continue
            
        # Skip immediate values, brackets, and complex patterns
        if re.match(r'^[#\[\]0-9\+\-\,\s\{\}]+$', token):
            continue
            
        # Keep ARM instructions and meaningful tokens (including short hex that could be opcodes)
        if (token in arm_instructions or 
            (len(token) >= 2 and len(token) <= 8 and 
             re.match(r'^[a-z0-9][a-z0-9]*$', token))):
            cleaned_tokens.append(token)
    
    return ' '.join(cleaned_tokens)

# step 3 - same CIG computation as original
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
        path = f'/home/josiah/research/IoT-Replication/IoTMalwareDetection-master/IoTMalwareDetection-master/Benign/all_goodware/{subdir}'
        benign_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.opcode')])

    print("==Cleaning benign opcode sequences==")
    with alive_bar(len(benign_files), title="Loading and cleaning benign samples") as bar:
        for filepath in benign_files:
            with open(filepath) as f:
                opcode_seq = f.read().replace('\n', ' ')
                cleaned_seq = clean_opcode_sequence(opcode_seq)
                if cleaned_seq.strip():  # Only add non-empty sequences
                    samples.append(cleaned_seq)
                    labels.append(0)  # benign = 0
            bar()

    # read malware opcodes with progress bar
    print("\n==Reading malware opcode files==")
    malware_path = '/home/josiah/research/IoT-Replication/IoTMalwareDetection-master/IoTMalwareDetection-master/Malware(Disassembeled)/'
    malware_files = [f for f in os.listdir(malware_path) if f.endswith('.opcode')]

    print("==Cleaning malware opcode sequences==")
    with alive_bar(len(malware_files), title="Loading and cleaning malware samples") as bar:
        for filename in os.listdir(malware_path):
            if filename.endswith('.opcode'):
                with open(os.path.join(malware_path, filename)) as f:
                    opcode_seq = f.read().replace('\n', ' ')
                    cleaned_seq = clean_opcode_sequence(opcode_seq)
                    if cleaned_seq.strip():  # Only add non-empty sequences
                        samples.append(cleaned_seq)
                        labels.append(1) # malware = 1
                bar()    

    # Show total samples
    print(f"\nTotal samples after cleaning: {len(samples)} \nBenign: {labels.count(0)} \nMalware: {labels.count(1)}\n")

    # Vectorization with better parameters
    print("==Vectorizing cleaned opcode sequences into n-grams (1-gram & 2-gram)==")
    with alive_bar(1, title="Vectorizing") as bar:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2), 
            token_pattern=r'\b[a-z0-9]+\b',
            min_df=2,  # Must appear in at least 2 documents
            max_df=0.95,  # Can't appear in more than 95% of documents
            max_features=10000  # Keep same limit as original
        )
        X = vectorizer.fit_transform(samples) # shape: (n_samples, n_features)
        bar()
    
    feature_names = vectorizer.get_feature_names_out()
    print(f"Feature vector shape: {X.shape} (samples x features)")
    
    # Show vocabulary stats
    unigrams = [f for f in feature_names if ' ' not in f]
    bigrams = [f for f in feature_names if ' ' in f]
    print(f"Vocabulary composition: {len(unigrams)} 1-grams, {len(bigrams)} 2-grams\n")

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

    # Analyze selected features
    selected_unigrams = [f for f in selected_feature_names if ' ' not in f]
    selected_bigrams = [f for f in selected_feature_names if ' ' in f]

    # Output Summary
    print(f"\nSelected top 82 features:")
    print(f"  1-grams: {len(selected_unigrams)}")
    print(f"  2-grams: {len(selected_bigrams)}")
    print(f"\nTop selected features:")
    print(selected_feature_names[:20])

    print(f"\nSelected 1-grams:")
    for unigram in selected_unigrams:
        print(f"  {unigram}")

    print(f"\nSelected 2-grams:")
    for bigram in selected_bigrams:
        print(f"  {bigram}")

    print(f"\nFinal selected feature matrix shape: {X_selected.shape}")

    # save data --> saves samples and selected_feature_names into a file using pickle format
    with open("improved_cig_output.pkl", "wb") as f:
        pickle.dump({
            "samples": samples,
            "selected_feature_names": selected_feature_names,
            "labels": labels,
            "vocabulary_stats": {
                "total_features": len(feature_names),
                "unigrams": len(unigrams),
                "bigrams": len(bigrams),
                "selected_unigrams": len(selected_unigrams),
                "selected_bigrams": len(selected_bigrams)
            }
        }, f)

    print("\nSaved improved CIG-selected features and samples to 'improved_cig_output.pkl'\n")

if __name__ == "__main__":
    main()
