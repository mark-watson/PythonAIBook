"""
Clustering example using K-Means on the Iris dataset.

Demonstrates unsupervised learning: discovering groups in data
without labeled targets.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def run_kmeans():
    """Run K-Means clustering on the Iris dataset."""
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names
    true_labels = iris.target
    species_names = iris.target_names

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Known species: {list(species_names)}\n")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try different values of k
    print("=== Silhouette Scores for Different k Values ===")
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k}: silhouette score = {score:.4f}")
    print()

    # Use k=3 (matching the 3 known species)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    # Compare clusters to true species labels
    print("=== K-Means (k=3) Cluster vs. True Species ===")
    df = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "true_species": [species_names[i] for i in true_labels],
        }
    )

    cross_tab = pd.crosstab(
        df["true_species"], df["cluster"], rownames=["Species"], colnames=["Cluster"]
    )
    print(cross_tab)
    print()

    # Cluster centers (in original scale)
    centers = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=feature_names)
    centers_df.index.name = "Cluster"
    print("=== Cluster Centers (original scale) ===")
    print(centers_df.round(2).to_string())
    print()


if __name__ == "__main__":
    run_kmeans()
