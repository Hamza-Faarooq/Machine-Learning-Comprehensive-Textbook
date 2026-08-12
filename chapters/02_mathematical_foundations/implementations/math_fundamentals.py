import numpy as np

class LinearAlgebraToolkit:
    """
    Mathematical foundations implemented from scratch using NumPy.
    """
    @staticmethod
    def pca_via_svd(X: np.ndarray, k: int) -> tuple:
        """
        Principal Component Analysis using Singular Value Decomposition[cite: 1].
        Returns the projected data and the principal components.
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute SVD: X = U \Sigma V^T
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Select top k components
        V_k = Vt[:k, :].T
        
        # Project the data
        Z = X_centered @ V_k
        
        # Compute variance explained
        eigenvalues = (Sigma ** 2.0) / (X.shape[0] - 1.0)
        variance_explained = np.sum(eigenvalues[:k]) / np.sum(eigenvalues)
        
        return Z, V_k, variance_explained

class InformationTheoryToolkit:
    """
    Information theory metrics for discrete distributions.
    """
    @staticmethod
    def shannon_entropy(p: np.ndarray) -> float:
        """
        Computes the Shannon entropy of a discrete distribution in bits[cite: 1].
        """
        # Filter out zero probabilities to avoid log(0)
        p_safe = p[p > 0.0]
        return float(-np.sum(p_safe * np.log2(p_safe)))

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Computes the Kullback-Leibler divergence D_KL(p || q)[cite: 1].
        """
        mask = (p > 0.0) & (q > 0.0)
        p_safe, q_safe = p[mask], q[mask]
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

if __name__ == "__main__":
    np.random.seed(42)
    
    print("--- PCA via SVD ---")
    X_dummy = np.random.randn(100, 5) # 100 samples, 5 features
    Z, components, var_exp = LinearAlgebraToolkit.pca_via_svd(X_dummy, k=2)
    print(f"Variance explained by top 2 components: {var_exp * 100.0:.1f}%")
    
    print("\n--- Information Theory ---")
    p_dist = np.array([0.5, 0.5])
    q_dist = np.array([0.8, 0.2])
    
    entropy = InformationTheoryToolkit.shannon_entropy(p_dist)
    kl_div = InformationTheoryToolkit.kl_divergence(p_dist, q_dist)
    
    print(f"Shannon Entropy of [0.5, 0.5]: {entropy:.1f} bits")
    print(f"KL Divergence D_KL(p||q):      {kl_div:.1f} nats")
