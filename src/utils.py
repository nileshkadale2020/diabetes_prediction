"""
Utility functions for plotting and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df, figsize=(10, 8)):
    """Plot correlation heatmap"""
    plt.figure(figsize=figsize)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt

def plot_feature_distributions(df, figsize=(15, 10)):
    """Plot distributions of all features"""
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.ravel()
    
    features = [col for col in df.columns if col != 'Outcome']
    
    for idx, feature in enumerate(features):
        if idx < len(axes):
            df[feature].hist(ax=axes[idx], bins=30, edgecolor='black')
            axes[idx].set_title(f'{feature} Distribution')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
    
    # Remove empty subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return plt

def plot_class_distribution(y, figsize=(8, 6)):
    """Plot class distribution"""
    plt.figure(figsize=figsize)
    class_counts = pd.Series(y).value_counts()
    plt.bar(['Non-Diabetic', 'Diabetic'], class_counts.values, color=['skyblue', 'salmon'])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    return plt

def save_plots(plt_obj, filename, directory='models'):
    """Save plot to file"""
    import os
    os.makedirs(directory, exist_ok=True)
    plt_obj.savefig(f'{directory}/{filename}', dpi=300, bbox_inches='tight')
    plt_obj.close()
    print(f"Plot saved to {directory}/{filename}")

