import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.decomposition import PCA
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clustering import get_kmeans_model, get_gmm_model

def evaluate_clustering(data_path='data/dataset_v1.parquet', zero_day_path='data/zero_day_dataset.parquet', seed=42):
    print("Loading datasets...")
    df = pd.read_parquet(data_path)
    X = df[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
    y = df['label'].values

    # We will just evaluate clustering on the main dataset for simplicity to find natural clusters of nominal, fim, twirl
    # Or we can combine with zero-day. Let's combine.
    if os.path.exists(zero_day_path):
        df_zd = pd.read_parquet(zero_day_path)
        X_zd = df_zd[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
        y_zd = df_zd['label'].values
        X_combined = np.vstack([X, X_zd])
        y_combined = np.concatenate([y, y_zd])
    else:
        X_combined = X
        y_combined = y

    # Subsample for speed in silhouette score if data is huge
    if len(X_combined) > 5000:
        np.random.seed(seed)
        idx = np.random.choice(len(X_combined), 5000, replace=False)
        X_sample = X_combined[idx]
        y_sample = y_combined[idx]
    else:
        X_sample = X_combined
        y_sample = y_combined

    print(f"Data shape for clustering evaluation: {X_sample.shape}")

    # Standardize
    scaler = QuantileTransformer(output_distribution='normal', random_state=seed)
    X_scaled = scaler.fit_transform(X_sample)

    k_values = list(range(2, 9))
    kmeans_silhouettes = []
    gmm_silhouettes = []
    gmm_bics = []

    print("Evaluating range of k...")
    for k in k_values:
        print(f"  k={k}...")
        # KMeans
        kmeans = get_kmeans_model(k, random_state=seed)
        kmeans_preds = kmeans.fit_predict(X_scaled)
        kmeans_silhouettes.append(silhouette_score(X_scaled, kmeans_preds))
        
        # GMM
        gmm = get_gmm_model(k, random_state=seed)
        gmm_preds = gmm.fit_predict(X_scaled)
        gmm_silhouettes.append(silhouette_score(X_scaled, gmm_preds))
        gmm_bics.append(gmm.bic(X_scaled))

    os.makedirs('figures', exist_ok=True)
    
    # Plot Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(k_values, kmeans_silhouettes, marker='o', label='K-Means')
    axes[0].plot(k_values, gmm_silhouettes, marker='o', label='GMM')
    axes[0].set_title('Silhouette Score vs. Number of Clusters (k)')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(k_values, gmm_bics, marker='o', color='red', label='GMM BIC')
    axes[1].set_title('Bayesian Information Criterion (BIC) vs. k')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('BIC (Lower is better)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('figures/clustering_metrics.png', dpi=300)
    print("Saved figures/clustering_metrics.png")

    # Evaluate at natural k. For our true labels, we expect k=4 (or 3 without zero-day). Let's evaluate at k=4.
    k_eval = 4
    print(f"\nDetailed Evaluation at k={k_eval} (Ground Truth k=4)...")
    gmm_final = get_gmm_model(k_eval, random_state=seed)
    preds = gmm_final.fit_predict(X_scaled)

    ari = adjusted_rand_score(y_sample, preds)
    nmi = normalized_mutual_info_score(y_sample, preds)
    v_measure = v_measure_score(y_sample, preds)

    print(f"GMM (k={k_eval}) - ARI: {ari:.4f}, NMI: {nmi:.4f}, V-Measure: {v_measure:.4f}")

    # Map clusters to majority ground truth for an "Accuracy" proxy
    def map_clusters(y_true, y_pred):
        mapped_preds = np.zeros_like(y_pred)
        for cluster in np.unique(y_pred):
            mask = (y_pred == cluster)
            majority_label = pd.Series(y_true[mask]).mode()[0]
            mapped_preds[mask] = majority_label
        return mapped_preds

    mapped_preds = map_clusters(y_sample, preds)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_sample, mapped_preds)
    print(f"GMM Majority-Vote Proxy Accuracy: {acc:.4f} (baseline supervised is ~0.88)")

    # PCA Visualization
    print("Generating PCA Visualization...")
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # True Labels
    scatter_true = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='viridis', alpha=0.5, s=10)
    axes[0].set_title('Ground Truth Classes (PCA)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    cbar1 = plt.colorbar(scatter_true, ax=axes[0])
    cbar1.set_label('True Class')

    # Predicted Clusters
    scatter_pred = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=preds, cmap='tab10', alpha=0.5, s=10)
    axes[1].set_title(f'GMM Clusters (k={k_eval}) (PCA)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    cbar2 = plt.colorbar(scatter_pred, ax=axes[1])
    cbar2.set_label('Cluster ID')

    plt.tight_layout()
    plt.savefig('figures/clustering_pca.png', dpi=300)
    print("Saved figures/clustering_pca.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset_v1.parquet')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    evaluate_clustering(args.data, seed=args.seed)
