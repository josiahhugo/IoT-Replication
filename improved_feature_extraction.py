#!/usr/bin/env python3
"""
Improved feature extraction to match paper results more closely
"""
import numpy as np
import pandas as pd
import os
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

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
        'msr', 'mrs', 'mul', 'mla', 'umull', 'umlal', 'smull', 'smlal'
    }
    
    for token in tokens:
        token = token.lower().strip()
        
        # Skip empty tokens
        if not token:
            continue
            
        # Skip pure hex addresses (like 0x1234, 00000000)
        if re.match(r'^(0x)?[0-9a-f]+$', token) and len(token) > 4:
            continue
            
        # Skip register names (r0, r1, etc.)
        if re.match(r'^r\d+$', token):
            continue
            
        # Skip special registers
        if token in ['sp', 'lr', 'pc', 'fp', 'ip', 'sl']:
            continue
            
        # Skip immediate values and offsets
        if re.match(r'^[#\[\]0-9\+\-\,\s]+$', token):
            continue
            
        # Keep ARM instructions and other meaningful tokens
        if (token in arm_instructions or 
            len(token) <= 8 and re.match(r'^[a-z][a-z0-9]*$', token)):
            cleaned_tokens.append(token)
    
    return ' '.join(cleaned_tokens)

def extract_features_like_paper():
    """
    Extract features using methodology closer to the paper
    """
    print("=== Improved Feature Extraction (Paper-like) ===\n")
    
    # Load data
    with open('/home/josiah/research/IoT-Replication/cig_output.pkl', 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    labels = data['labels']
    
    print(f"Original samples: {len(samples)}")
    
    # Clean the opcode sequences
    print("Cleaning opcode sequences...")
    cleaned_samples = []
    for i, sample in enumerate(samples):
        cleaned = clean_opcode_sequence(sample)
        if cleaned.strip():  # Only keep non-empty cleaned samples
            cleaned_samples.append(cleaned)
        else:
            print(f"Warning: Sample {i} became empty after cleaning")
    
    print(f"Cleaned samples: {len(cleaned_samples)}")
    
    # Extract features without max_features limit first
    print("\n=== Full Vocabulary Analysis ===")
    vectorizer_full = CountVectorizer(
        ngram_range=(1, 2), 
        token_pattern=r'\b[a-z][a-z0-9]*\b',  # Only alphabetic tokens
        min_df=2,  # Minimum document frequency (appears in at least 2 samples)
        max_df=0.95  # Maximum document frequency (appears in at most 95% of samples)
    )
    
    X_full = vectorizer_full.fit_transform(cleaned_samples)
    feature_names_full = vectorizer_full.get_feature_names_out()
    
    # Separate 1-grams and 2-grams
    unigrams_full = [f for f in feature_names_full if ' ' not in f]
    bigrams_full = [f for f in feature_names_full if ' ' in f]
    
    print(f"After cleaning and filtering:")
    print(f"  Total distinct 1-grams: {len(unigrams_full):,}")
    print(f"  Total distinct 2-grams: {len(bigrams_full):,}")
    print(f"  Total features: {len(feature_names_full):,}")
    
    print(f"\nComparison with paper:")
    print(f"  Paper: 4,543 1-grams, 610,109 2-grams")
    print(f"  Cleaned: {len(unigrams_full):,} 1-grams, {len(bigrams_full):,} 2-grams")
    
    # Show sample features
    print(f"\nSample cleaned 1-grams:")
    print(unigrams_full[:20])
    
    print(f"\nSample cleaned 2-grams:")
    print(bigrams_full[:20])
    
    # Get frequency distribution
    feature_freq = np.array(X_full.sum(axis=0)).flatten()
    freq_df = pd.DataFrame({
        'feature': feature_names_full,
        'frequency': feature_freq,
        'type': ['2-gram' if ' ' in f else '1-gram' for f in feature_names_full]
    })
    
    print(f"\nTop 20 most frequent 1-grams:")
    top_unigrams = freq_df[freq_df['type'] == '1-gram'].nlargest(20, 'frequency')
    for _, row in top_unigrams.iterrows():
        print(f"  {row['feature']}: {row['frequency']}")
    
    print(f"\nTop 20 most frequent 2-grams:")
    top_bigrams = freq_df[freq_df['type'] == '2-gram'].nlargest(20, 'frequency')
    for _, row in top_bigrams.iterrows():
        print(f"  {row['feature']}: {row['frequency']}")
    
    # Save cleaned data
    cleaned_data = {
        'cleaned_samples': cleaned_samples,
        'labels': labels[:len(cleaned_samples)],  # Match the length
        'vocabulary_size': len(feature_names_full),
        'unigrams': len(unigrams_full),
        'bigrams': len(bigrams_full)
    }
    
    with open('/home/josiah/research/IoT-Replication/cleaned_features.pkl', 'wb') as f:
        pickle.dump(cleaned_data, f)
    
    print(f"\nSaved cleaned data to 'cleaned_features.pkl'")
    
    return cleaned_samples, labels[:len(cleaned_samples)]

if __name__ == "__main__":
    extract_features_like_paper()
