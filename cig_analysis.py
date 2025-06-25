#!/usr/bin/env python3
"""
Re-run CIG computation to see the actual scores for 1-grams vs 2-grams
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def compute_cig_simplified(X, y): 
    """Simplified CIG computation to get scores"""
    n_samples, n_features = X.shape
    y = np.array(y)
    cig_scores = []

    # Class priors
    p_benign = np.mean(y == 0)
    p_malware = np.mean(y == 1)

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

        total_cig = cig_benign + cig_malware
        cig_scores.append((j, total_cig))

    return cig_scores

def analyze_cig_scores():
    print("=== Analyzing CIG Scores for 1-grams vs 2-grams ===\n")
    
    try:
        with open('/home/josiah/research/IoT-Replication/cig_output.pkl', 'rb') as f:
            data = pickle.load(f)
        
        samples = data['samples']
        labels = data['labels']
        
        print("Re-running vectorization and CIG computation...")
        
        # First, check total distinct n-grams without max_features limit
        print("\n=== Checking Total Distinct N-grams (No Limit) ===")
        vectorizer_full = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b')
        X_full = vectorizer_full.fit_transform(samples)
        feature_names_full = vectorizer_full.get_feature_names_out()
        
        # Separate 1-grams and 2-grams
        unigrams_full = [f for f in feature_names_full if ' ' not in f]
        bigrams_full = [f for f in feature_names_full if ' ' in f]
        
        print(f"Total distinct 1-grams: {len(unigrams_full)}")
        print(f"Total distinct 2-grams: {len(bigrams_full)}")
        print(f"Total features: {len(feature_names_full)}")
        
        # Compare with paper results
        print(f"\nComparison with paper:")
        print(f"Paper: 4,543 1-grams, 610,109 2-grams")
        print(f"Yours: {len(unigrams_full):,} 1-grams, {len(bigrams_full):,} 2-grams")
        
        # Now use the same parameters as your original CIG computation
        print(f"\n=== Using Original Parameters (max_features=10000) ===")
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', max_features=10000)
        X = vectorizer.fit_transform(samples)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"Computing CIG for {len(feature_names)} features...")
        cig_scores = compute_cig_simplified(X, labels)
        
        # Create dataframe
        df = pd.DataFrame({
            'feature': feature_names,
            'cig_score': [score for _, score in cig_scores],
            'type': ['2-gram' if ' ' in f else '1-gram' for f in feature_names]
        })
        
        # Sort by CIG score
        df = df.sort_values('cig_score', ascending=False)
        
        print(f"\nCIG Score Statistics:")
        print(df.groupby('type')['cig_score'].describe())
        
        print(f"\nTop 20 features by CIG score:")
        print(df.head(20)[['feature', 'type', 'cig_score']])
        
        print(f"\nTop 10 2-grams by CIG score:")
        top_bigrams = df[df['type'] == '2-gram'].head(10)
        print(top_bigrams[['feature', 'cig_score']])
        
        print(f"\nTop 10 1-grams by CIG score:")
        top_unigrams = df[df['type'] == '1-gram'].head(10)
        print(top_unigrams[['feature', 'cig_score']])
        
        # Check if push str is in our top features
        push_str_row = df[df['feature'] == 'push str']
        if not push_str_row.empty:
            rank = push_str_row.index[0] + 1
            score = push_str_row['cig_score'].iloc[0]
            print(f"\n'push str' rank: {rank}, CIG score: {score:.6f}")
        
        # Check distribution in top 82
        top_82 = df.head(82)
        type_counts = top_82['type'].value_counts()
        print(f"\nIn top 82 features:")
        print(f"1-grams: {type_counts.get('1-gram', 0)}")
        print(f"2-grams: {type_counts.get('2-gram', 0)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_cig_scores()
