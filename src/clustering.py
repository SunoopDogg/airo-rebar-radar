"""Clustering module for rebar point separation."""

import numpy as np
from sklearn.cluster import DBSCAN

from .utils.config import ClusteringConfig


class Clusterer:
    """Cluster LIDAR points into individual rebar groups."""

    def __init__(self, config: ClusteringConfig | None = None):
        """Initialize clusterer.

        Args:
            config: Clustering configuration
        """
        self.config = config or ClusteringConfig()
        self._dbscan = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples
        )

    def cluster(self, points: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Cluster points using DBSCAN.

        Args:
            points: Array of shape (N, 2) with x, y coordinates

        Returns:
            Tuple of (labels, cluster_list) where:
                - labels: Array of cluster labels (-1 for noise)
                - cluster_list: List of point arrays, one per valid cluster
        """
        if len(points) == 0:
            return np.array([]), []

        # Fit DBSCAN
        labels = self._dbscan.fit_predict(points)

        # Extract individual clusters (excluding noise points labeled -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        clusters = []
        for label in sorted(unique_labels):
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]
            clusters.append(cluster_points)

        return labels, clusters

    def get_cluster_stats(
        self,
        labels: np.ndarray,
        points: np.ndarray
    ) -> dict:
        """Calculate statistics about clustering results.

        Args:
            labels: Cluster labels from DBSCAN
            points: Original points

        Returns:
            Dictionary with clustering statistics
        """
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = int(np.sum(labels == -1))

        # Use bincount for O(n) instead of O(n*k) loop
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            cluster_sizes = list(np.bincount(valid_labels))
        else:
            cluster_sizes = []

        return {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "n_total_points": len(points),
            "cluster_sizes": cluster_sizes,
            "mean_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
        }
