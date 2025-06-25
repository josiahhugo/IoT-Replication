"""
Feature Analysis and Visualization for CIG-Selected Features
Compares the top 82 features between malware and benign samples
Creates visualizations similar to research papers
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_cig_results():
    """Load the CIG results from pickle file"""
    with open("cig_output.pkl", "rb") as f:
        data = pickle.load(f)
    return data

def analyze_feature_distribution(samples, labels, selected_features):
    """Analyze the distribution of selected features in malware vs benign samples"""
    
    # Create vectorizer with only selected features
    vectorizer = CountVectorizer(
        ngram_range=(1, 2), 
        token_pattern=r'\b\w+\b',
        vocabulary=selected_features
    )
    
    X = vectorizer.fit_transform(samples)
    feature_names = vectorizer.get_feature_names_out()
    
    # Separate malware and benign samples
    malware_indices = [i for i, label in enumerate(labels) if label == 1]
    benign_indices = [i for i, label in enumerate(labels) if label == 0]
    
    # Calculate feature frequencies
    malware_freq = np.array(X[malware_indices].sum(axis=0)).flatten()
    benign_freq = np.array(X[benign_indices].sum(axis=0)).flatten()
    
    # Normalize by sample count
    malware_freq_norm = malware_freq / len(malware_indices)
    benign_freq_norm = benign_freq / len(benign_indices)
    
    # Create dataframe for analysis
    df = pd.DataFrame({
        'feature': feature_names,
        'malware_freq': malware_freq,
        'benign_freq': benign_freq,
        'malware_freq_norm': malware_freq_norm,
        'benign_freq_norm': benign_freq_norm,
        'ratio': (malware_freq_norm + 1e-9) / (benign_freq_norm + 1e-9),
        'is_bigram': [' ' in feat for feat in feature_names]
    })
    
    return df

def create_feature_comparison_plot(df, top_n=30):
    """Create a comparison plot of top features"""
    
    # Sort by ratio (malware preference)
    df_sorted = df.nlargest(top_n, 'ratio')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Feature frequency comparison (log scale)
    x_pos = np.arange(len(df_sorted))
    width = 0.35
    
    ax1.bar(x_pos - width/2, df_sorted['malware_freq_norm'], width, 
            label='Malware', alpha=0.8, color='red')
    ax1.bar(x_pos + width/2, df_sorted['benign_freq_norm'], width, 
            label='Benign', alpha=0.8, color='blue')
    
    ax1.set_xlabel('Features (Top 30 by Malware/Benign Ratio)')
    ax1.set_ylabel('Normalized Frequency')
    ax1.set_title('Feature Frequency Distribution: Malware vs Benign')
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_sorted['feature'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio plot
    colors = ['orange' if is_bigram else 'green' for is_bigram in df_sorted['is_bigram']]
    bars = ax2.bar(x_pos, df_sorted['ratio'], color=colors, alpha=0.7)
    
    ax2.set_xlabel('Features (Top 30 by Malware/Benign Ratio)')
    ax2.set_ylabel('Malware/Benign Frequency Ratio (log scale)')
    ax2.set_title('Feature Selectivity: Higher Values Indicate Malware Preference')
    ax2.set_yscale('log')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_sorted['feature'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='1-gram'),
                      Patch(facecolor='orange', alpha=0.7, label='2-gram')]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('feature_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def create_feature_heatmap(df, top_n=50):
    """Create a heatmap showing feature importance"""
    
    df_top = df.nlargest(top_n, 'ratio')
    
    # Prepare data for heatmap
    data_matrix = np.array([df_top['malware_freq_norm'].values, 
                           df_top['benign_freq_norm'].values])
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(df_top)))
    ax.set_xticklabels(df_top['feature'], rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Malware', 'Benign'])
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Normalized Frequency')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(df_top)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'Top {top_n} CIG-Selected Features: Frequency Heatmap')
    plt.tight_layout()
    plt.savefig('feature_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def create_ngram_distribution_plot(df):
    """Create a plot showing 1-gram vs 2-gram distribution"""
    
    unigrams = df[~df['is_bigram']]
    bigrams = df[df['is_bigram']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Count of 1-grams vs 2-grams
    counts = [len(unigrams), len(bigrams)]
    labels = ['1-grams', '2-grams']
    colors = ['skyblue', 'lightcoral']
    
    ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribution of Selected Features by N-gram Type')
    
    # Plot 2: Top 1-grams by ratio
    top_unigrams = unigrams.nlargest(15, 'ratio')
    ax2.barh(range(len(top_unigrams)), top_unigrams['ratio'], color='skyblue', alpha=0.7)
    ax2.set_yticks(range(len(top_unigrams)))
    ax2.set_yticklabels(top_unigrams['feature'])
    ax2.set_xlabel('Malware/Benign Ratio')
    ax2.set_title('Top 15 1-grams by Malware Preference')
    ax2.set_xscale('log')
    
    # Plot 3: Top 2-grams by ratio
    top_bigrams = bigrams.nlargest(15, 'ratio')
    ax3.barh(range(len(top_bigrams)), top_bigrams['ratio'], color='lightcoral', alpha=0.7)
    ax3.set_yticks(range(len(top_bigrams)))
    ax3.set_yticklabels(top_bigrams['feature'])
    ax3.set_xlabel('Malware/Benign Ratio')
    ax3.set_title('Top 15 2-grams by Malware Preference')
    ax3.set_xscale('log')
    
    # Plot 4: Average frequencies
    avg_data = {
        'Feature Type': ['1-grams', '2-grams', '1-grams', '2-grams'],
        'Sample Type': ['Malware', 'Malware', 'Benign', 'Benign'],
        'Average Frequency': [
            unigrams['malware_freq_norm'].mean(),
            bigrams['malware_freq_norm'].mean(),
            unigrams['benign_freq_norm'].mean(),
            bigrams['benign_freq_norm'].mean()
        ]
    }
    
    df_avg = pd.DataFrame(avg_data)
    sns.barplot(data=df_avg, x='Feature Type', y='Average Frequency', hue='Sample Type', ax=ax4)
    ax4.set_title('Average Feature Frequencies by Type and Sample Class')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ngram_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def print_feature_summary(df):
    """Print a summary of the selected features"""
    
    print("="*80)
    print("CIG FEATURE SELECTION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Total selected features: {len(df)}")
    print(f"1-grams: {len(df[~df['is_bigram']])}")
    print(f"2-grams: {len(df[df['is_bigram']])}")
    
    print("\n" + "="*50)
    print("TOP 20 FEATURES BY MALWARE PREFERENCE")
    print("="*50)
    
    top_20 = df.nlargest(20, 'ratio')
    for i, (_, row) in enumerate(top_20.iterrows(), 1):
        feature_type = "2-gram" if row['is_bigram'] else "1-gram"
        print(f"{i:2d}. {row['feature']:<20} ({feature_type:<6}) - Ratio: {row['ratio']:.3f}")
    
    print("\n" + "="*50)
    print("FEATURE TYPE ANALYSIS")
    print("="*50)
    
    unigrams = df[~df['is_bigram']]
    bigrams = df[df['is_bigram']]
    
    print(f"Average malware frequency (1-grams): {unigrams['malware_freq_norm'].mean():.4f}")
    print(f"Average malware frequency (2-grams): {bigrams['malware_freq_norm'].mean():.4f}")
    print(f"Average benign frequency (1-grams): {unigrams['benign_freq_norm'].mean():.4f}")
    print(f"Average benign frequency (2-grams): {bigrams['benign_freq_norm'].mean():.4f}")
    
    print(f"\nMost discriminative 1-gram: {unigrams.loc[unigrams['ratio'].idxmax(), 'feature']}")
    print(f"Most discriminative 2-gram: {bigrams.loc[bigrams['ratio'].idxmax(), 'feature']}")

def main():
    """Main function to run the complete analysis"""
    
    print("Loading CIG results...")
    data = load_cig_results()
    
    samples = data['samples']
    labels = data['labels']
    selected_features = data['selected_feature_names']
    
    print(f"Loaded {len(samples)} samples with {len(selected_features)} selected features")
    
    print("Analyzing feature distribution...")
    df = analyze_feature_distribution(samples, labels, selected_features)
    
    print("Creating visualizations...")
    create_feature_comparison_plot(df)
    create_feature_heatmap(df)
    create_ngram_distribution_plot(df)
    
    print("Generating summary...")
    print_feature_summary(df)
    
    # Save the analysis results
    df.to_csv('cig_feature_analysis.csv', index=False)
    print("\nAnalysis results saved to 'cig_feature_analysis.csv'")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()
