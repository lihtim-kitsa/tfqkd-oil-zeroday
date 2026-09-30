from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def get_kmeans_model(n_clusters, random_state=42):
    """
    Returns a KMeans model.
    """
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

def get_gmm_model(n_components, random_state=42):
    """
    Returns a Gaussian Mixture Model.
    """
    return GaussianMixture(n_components=n_components, random_state=random_state)
