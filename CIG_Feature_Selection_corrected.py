'''
Corrected CIG Feature Selection focused on common ARM instructions
This version should produce features that co-occur more frequently, 
leading to denser graphs for eigenspace transformation.
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

samples = []
labels = []

def clean_opcode_sequence(opcode_sequence):
    """
    Clean opcode sequence focusing on core ARM instructions that appear frequently
    This should produce features with better co-occurrence for graph construction
    """
    tokens = opcode_sequence.split()
    cleaned_tokens = []
    
    # Core ARM instructions (base forms without conditional suffixes)
    # Focus on the most common instructions that would co-occur frequently
    core_arm_instructions = {
        # Data movement
        'mov', 'ldr', 'str', 'push', 'pop',
        'ldrb', 'strb', 'ldrh', 'strh',
        
        # Arithmetic & Logic  
        'add', 'sub', 'cmp', 'and', 'orr', 'eor', 'bic', 'mvn',
        'mul', 'tst',
        
        # Control flow
        'bl', 'blx', 'b', 'bx',
        
        # Common conditional variants (but limit to most frequent)
        'beq', 'bne', 'blt', 'bgt', 'ble', 'bge',
        
        # Shift operations  
        'lsl', 'lsr', 'asr',
        
        # Multiple load/store
        'ldm', 'stm',
        
        # Stack operations are important for malware analysis
        'push', 'pop'
    }
    
    for token in tokens:
        token = token.lower().strip()
        
        # Skip empty tokens
        if not token:
            continue
            
        # Skip hex addresses longer than 6 chars (likely addresses, not opcodes)
        if re.match(r'^(0x)?[0-9a-f]+$', token) and len(token) > 6:
            continue
            
        # Skip register names
        if re.match(r'^r\d+$', token):
            continue
            
        # Skip special registers  
        if token in ['sp', 'lr', 'pc', 'fp', 'ip', 'sl']:
            continue
            
        # Skip immediate values, brackets, punctuation
        if re.match(r'^[#\[\]0-9\+\-\,\s\{\}\(\)\.]+$', token):
            continue
            
        # Extract base instruction from conditional variants
        # e.g., "moveq" -> "mov", "ldrne" -> "ldr"  
        base_instruction = token
        for suffix in ['eq', 'ne', 'lt', 'gt', 'le', 'ge', 'ls', 'hi', 'cs', 'cc', 'mi', 'pl', 'vs', 'vc', 's']:
            if token.endswith(suffix) and len(token) > len(suffix):
                potential_base = token[:-len(suffix)]
                if potential_base in core_arm_instructions:
                    base_instruction = potential_base
                    break
        
        # Keep only core ARM instructions
        if base_instruction in core_arm_instructions:
            cleaned_tokens.append(base_instruction)
        # Also keep some common conditional variants that are important for control flow
        elif token in ['beq', 'bne', 'blt', 'bgt', 'ble', 'bge', 'bls', 'bhi']:
            cleaned_tokens.append(token)
    
    return ' '.join(cleaned_tokens)

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

            # Feature value distribution
            unique_values = np.unique(v)
            
            cig_j = 0.0
            
            for val in unique_values:
                # P(F_j = val)
                p_val = np.mean(v == val)
                
                if p_val == 0:
                    continue
                
                # P(F_j = val | C = benign)
                benign_indices = (y == 0)
                if np.sum(benign_indices) > 0:
                    p_val_given_benign = np.mean(v[benign_indices] == val)
                else:
                    p_val_given_benign = 0
                
                # P(F_j = val | C = malware)
                malware_indices = (y == 1)
                if np.sum(malware_indices) > 0:
                    p_val_given_malware = np.mean(v[malware_indices] == val)
                else:
                    p_val_given_malware = 0
                
                # CIG contribution for this value
                if p_val_given_benign > 0 and p_val_given_malware > 0:
                    term1 = p_val_given_benign * p_benign * np.log2(p_val_given_benign / p_val)
                    term2 = p_val_given_malware * p_malware * np.log2(p_val_given_malware / p_val)
                    cig_j += (term1 + term2)
            
            cig_scores.append(cig_j)
            bar()

    return cig_scores

def main():
    # Read benign opcodes
    print("==Reading benign opcode files==")
    benign_dir = "IoTMalwareDetection-master/IoTMalwareDetection-master/Benign/all_goodware"
    
    for subdir in os.listdir(benign_dir):
        subdir_path = os.path.join(benign_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.opcode'):
                    file_path = os.path.join(subdir_path, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            cleaned_content = clean_opcode_sequence(content)
                            if cleaned_content.strip():  # Only add non-empty sequences
                                samples.append(cleaned_content)
                                labels.append(0)  # benign
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

    print(f"Loaded {sum(1 for l in labels if l == 0)} benign samples")

    # Read malware opcodes
    print("==Reading malware opcode files==")
    malware_dir = "IoTMalwareDetection-master/IoTMalwareDetection-master/Malware(Disassembeled)"
    
    for file in os.listdir(malware_dir):
        if file.endswith('.opcode'):
            file_path = os.path.join(malware_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    cleaned_content = clean_opcode_sequence(content)
                    if cleaned_content.strip():  # Only add non-empty sequences
                        samples.append(cleaned_content)
                        labels.append(1)  # malware
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

    print(f"Loaded {sum(1 for l in labels if l == 1)} malware samples")
    print(f"Total samples: {len(samples)}")

    # Create n-gram features with better parameters
    print("\n==Creating N-gram Feature Matrix==")
    
    # Create separate vectorizers for 1-grams and 2-grams to ensure good balance
    vectorizer_1gram = CountVectorizer(
        ngram_range=(1, 1),  # Only 1-grams
        min_df=10,          # 1-grams should appear in at least 10 samples
        max_df=0.7,         # Don't select overly common 1-grams 
        max_features=500,   # Limit 1-gram features
        token_pattern=r'\b\w+\b'
    )
    
    vectorizer_2gram = CountVectorizer(
        ngram_range=(2, 2),  # Only 2-grams
        min_df=5,           # 2-grams can be less frequent
        max_df=0.6,         # 2-grams should be discriminative
        max_features=1500,  # More 2-gram features
        token_pattern=r'\b\w+\b'
    )
    
    # Get 1-gram and 2-gram features separately
    X_1gram = vectorizer_1gram.fit_transform(samples)
    X_2gram = vectorizer_2gram.fit_transform(samples)
    
    # Combine feature matrices
    from scipy.sparse import hstack
    X = hstack([X_1gram, X_2gram])
    
    feature_names_1gram = vectorizer_1gram.get_feature_names_out()
    feature_names_2gram = vectorizer_2gram.get_feature_names_out()
    feature_names = np.concatenate([feature_names_1gram, feature_names_2gram])
    
    
    print(f"Initial feature matrix shape: {X.shape}")
    print(f"1-gram features: {len(feature_names_1gram)}")
    print(f"2-gram features: {len(feature_names_2gram)}")
    print(f"Example 1-grams: {feature_names_1gram[:10]}")
    print(f"Example 2-grams: {feature_names_2gram[:10]}")

    # Compute CIG scores
    print("\n==Computing CIG Scores==")
    cig_scores = compute_cig(X, labels)

    # Select top 82 features
    print("\n==Selecting Top 82 Features==")
    top_indices = np.argsort(cig_scores)[::-1][:82]
    selected_feature_names = [feature_names[i] for i in top_indices]
    selected_cig_scores = [cig_scores[i] for i in top_indices]

    X_selected = X[:, top_indices]

    print(f"Selected {len(selected_feature_names)} features")
    print(f"Top 10 features:")
    for i in range(10):
        print(f"{i+1:2d}. {selected_feature_names[i]} (CIG: {selected_cig_scores[i]:.6f})")

    # Analyze feature types
    unigrams = [f for f in selected_feature_names if ' ' not in f]
    bigrams = [f for f in selected_feature_names if ' ' in f]
    print(f"\n1-grams: {len(unigrams)}, 2-grams: {len(bigrams)}")

    # Save results
    print("\n==Saving Results==")
    output_data = {
        'samples': samples,
        'labels': labels,
        'X_selected': X_selected,
        'selected_feature_names': selected_feature_names,
        'cig_scores': selected_cig_scores,
        'feature_names_all': feature_names,
        'cig_scores_all': cig_scores
    }

    with open("corrected_cig_output.pkl", "wb") as f:
        pickle.dump(output_data, f)

    print("Saved corrected feature selection to 'corrected_cig_output.pkl'")

    # Create visualization of selected features
    plt.figure(figsize=(12, 8))
    plt.barh(range(20), selected_cig_scores[:20])
    plt.yticks(range(20), [f[:30] for f in selected_feature_names[:20]])
    plt.xlabel('CIG Score')
    plt.title('Top 20 Selected Features by CIG Score')
    plt.tight_layout()
    plt.savefig('corrected_feature_selection.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved feature visualization to 'corrected_feature_selection.png'")

if __name__ == "__main__":
    main()
