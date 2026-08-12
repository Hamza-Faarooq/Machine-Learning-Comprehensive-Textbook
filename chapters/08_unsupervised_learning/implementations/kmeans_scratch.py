import numpy as np
from typing import Tuple

class KMeansScratch:
    """
    k-Means clustering via Lloyd's algorithm from scratch[cite: 1].
    """
    def __init__(self, n_clusters: int = 3, max_iters: int = 100) -> None:
        self.k = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alternates between assigning points to the nearest centroid and 
        updating centroids to the mean of assigned points[cite: 1].
        """
        n_samples, n_features = X.shape
        
        # Random Initialization
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        labels = np.zeros(n_samples)
        
        for _ in range(self.max_iters):
            # E-step: Assign each point to the closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)
            
            # Check for convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # M-step: Recompute centroids
            for cluster_idx in range(self.k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) > 0:
                    self.centroids[cluster_idx] = np.mean(cluster_points, axis=0)
                    
        return labels, self.centroids

if __name__ == "__main__":
    np.random.seed(42)
    # Simple 2D blobs
    X_dummy = np.vstack((
        np.random.randn(50, 2) + np.array([5.0, 5.0]),
        np.random.randn(50, 2) + np.array([-5.0, -5.0])
    ))
    
    kmeans = KMeansScratch(n_clusters=2)
    labels, centers = kmeans.fit(X_dummy)
    
    print("--- k-Means Clustering ---")
    print(f"Centroid 1: [{centers[0][0]:.1f}, {centers[0][1]:.1f}]")
    print(f"Centroid 2: [{centers[1][0]:.1f}, {centers[1][1]:.1f}]")
