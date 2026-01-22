"""
Standalone script to analyze series embeddings after training.

Usage:
    python scripts/analyze_embeddings.py --checkpoint path/to/checkpoint.ckpt
    
Or use interactively:
    python
    >>> from scripts.analyze_embeddings import analyze_model
    >>> analyze_model('path/to/checkpoint.ckpt')
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.models import AnyQuantileWithSeriesEmbedding


# Default European country names for MHLV dataset
EUROPEAN_COUNTRIES = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus',
    'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',
    'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
    'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
    'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden', 'United Kingdom', 'Norway', 'Switzerland',
    'Serbia', 'Bosnia', 'Macedonia', 'Montenegro', 'Albania'
]


def load_model_from_checkpoint(checkpoint_path):
    """Load model from PyTorch Lightning checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # Try loading as Lightning checkpoint
        model = AnyQuantileWithSeriesEmbedding.load_from_checkpoint(checkpoint_path)
        print("✓ Loaded as PyTorch Lightning checkpoint")
    except Exception as e:
        print(f"Could not load as Lightning checkpoint: {e}")
        try:
            # Try loading as state dict
            checkpoint = torch.load(checkpoint_path)
            print(f"Checkpoint keys: {checkpoint.keys()}")
            # You may need to instantiate model with config first
            raise NotImplementedError("Please load model with config first")
        except Exception as e2:
            print(f"Error loading checkpoint: {e2}")
            raise
    
    model.eval()
    return model


def analyze_series_embeddings(model, country_names=None, save_path='./embedding_analysis'):
    """
    Comprehensive analysis of learned series embeddings.
    
    Args:
        model: Trained model with series_embedding attribute
        country_names: List of country names
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Check if model has series embeddings
    if not hasattr(model, 'series_embedding'):
        print("❌ Model does not have series_embedding attribute!")
        print("   Make sure you're using AnyQuantileWithSeriesEmbedding")
        return None
    
    # Extract embeddings
    embeddings = model.series_embedding.weight.detach().cpu().numpy()
    num_series, embed_dim = embeddings.shape
    
    if country_names is None:
        country_names = EUROPEAN_COUNTRIES[:num_series]
    elif len(country_names) != num_series:
        print(f"⚠️  Warning: {len(country_names)} names provided but {num_series} series in model")
        country_names = country_names[:num_series]
    
    print(f"\n{'='*70}")
    print(f"{'SERIES EMBEDDING ANALYSIS':^70}")
    print(f"{'='*70}")
    print(f"Number of series: {num_series}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Total embedding parameters: {num_series * embed_dim:,}")
    
    # 1. Similarity Analysis
    print(f"\n{'1. SIMILARITY ANALYSIS':-^70}")
    similarity = model.get_series_similarity().detach().cpu().numpy()
    
    # Find top similar pairs
    similarity_no_diag = similarity.copy()
    np.fill_diagonal(similarity_no_diag, -1)
    
    print("\nTop 10 most similar country pairs:")
    similar_pairs = []
    for i in range(num_series):
        j = similarity_no_diag[i].argmax()
        sim_score = similarity[i, j]
        if i < j:  # Avoid duplicates
            similar_pairs.append((country_names[i], country_names[j], sim_score, i, j))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    for rank, (name_i, name_j, score, i, j) in enumerate(similar_pairs[:10], 1):
        print(f"  {rank:2d}. {name_i:18s} ↔ {name_j:18s}  similarity: {score:.4f}")
    
    # Plot similarity heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(similarity, dtype=bool), k=1)  # Mask upper triangle
    sns.heatmap(similarity, 
                mask=mask,
                xticklabels=country_names,
                yticklabels=country_names,
                cmap='RdYlBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'},
                annot=False,
                fmt='.2f',
                linewidths=0.5)
    plt.title('Learned Country Similarity Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}/similarity_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}/similarity_heatmap.png")
    plt.close()
    
    # 2. t-SNE Visualization
    print(f"\n{'2. t-SNE VISUALIZATION':-^70}")
    try:
        from sklearn.manifold import TSNE
        perplexity = min(30, num_series - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            s=200, alpha=0.6, c=range(num_series), cmap='tab20')
        
        for i, name in enumerate(country_names):
            plt.annotate(name, 
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.title('t-SNE: Country Embedding Space', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/tsne_visualization.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}/tsne_visualization.png")
        plt.close()
    except ImportError:
        print("⚠️  sklearn not available, skipping t-SNE")
        embeddings_2d = None
    
    # 3. PCA Analysis
    print(f"\n{'3. PCA ANALYSIS':-^70}")
    try:
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(embeddings)
        explained_variance = pca.explained_variance_ratio_
        
        n_components_90 = np.argmax(np.cumsum(explained_variance) >= 0.90) + 1
        n_components_95 = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
        
        print(f"Components needed for 90% variance: {n_components_90}/{embed_dim}")
        print(f"Components needed for 95% variance: {n_components_95}/{embed_dim}")
        print(f"Top 3 components explain: {np.sum(explained_variance[:3])*100:.1f}% of variance")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cumulative variance
        axes[0].plot(range(1, len(explained_variance)+1), 
                    np.cumsum(explained_variance), 'o-', linewidth=2, markersize=6)
        axes[0].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
        axes[0].axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
        axes[0].set_xlabel('Number of Components', fontsize=11)
        axes[0].set_ylabel('Cumulative Explained Variance', fontsize=11)
        axes[0].set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Individual variance
        n_show = min(15, embed_dim)
        axes[1].bar(range(1, n_show+1), explained_variance[:n_show])
        axes[1].set_xlabel('Principal Component', fontsize=11)
        axes[1].set_ylabel('Explained Variance Ratio', fontsize=11)
        axes[1].set_title('Individual Component Variance', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/pca_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}/pca_analysis.png")
        plt.close()
    except ImportError:
        print("⚠️  sklearn not available, skipping PCA")
        pca = None
    
    # 4. Embedding Statistics
    print(f"\n{'4. EMBEDDING STATISTICS':-^70}")
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norm: {embedding_norms.mean():.4f} ± {embedding_norms.std():.4f}")
    print(f"Min/Max norm: {embedding_norms.min():.4f} / {embedding_norms.max():.4f}")
    
    # Check for potential issues
    norm_cv = embedding_norms.std() / embedding_norms.mean()
    if norm_cv > 0.5:
        print(f"⚠️  High coefficient of variation ({norm_cv:.2f}) in embedding norms")
        print("   Some countries may dominate - consider normalization")
    else:
        print(f"✓ Reasonable variation in embedding norms (CV={norm_cv:.2f})")
    
    # 5. Clustering
    print(f"\n{'5. CLUSTERING ANALYSIS':-^70}")
    try:
        from sklearn.cluster import KMeans
        
        for k in [3, 4, 5]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            print(f"\nK={k} Clustering:")
            for cluster_id in range(k):
                members = [country_names[i] for i in range(num_series) if clusters[i] == cluster_id]
                print(f"  Cluster {cluster_id+1}: {', '.join(members)}")
    except ImportError:
        print("⚠️  sklearn not available, skipping clustering")
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {save_path}/")
    print(f"{'='*70}\n")
    
    return {
        'embeddings': embeddings,
        'similarity': similarity,
        'tsne_coords': embeddings_2d,
        'embedding_norms': embedding_norms,
        'country_names': country_names
    }


def analyze_model(checkpoint_path, country_names=None, save_path='./embedding_analysis'):
    """Convenience function to load checkpoint and run analysis."""
    model = load_model_from_checkpoint(checkpoint_path)
    return analyze_series_embeddings(model, country_names, save_path)


def main():
    parser = argparse.ArgumentParser(description='Analyze learned series embeddings')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./embedding_analysis',
                       help='Output directory for plots')
    parser.add_argument('--countries', type=str, nargs='+', default=None,
                       help='Country names (optional)')
    
    args = parser.parse_args()
    
    analyze_model(args.checkpoint, args.countries, args.output)


if __name__ == '__main__':
    main()