"""Preprocessing module for LIDAR point cloud data."""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .utils.config import PreprocessingConfig


class Preprocessor:
    """Preprocess LIDAR point cloud data."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def filter_by_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter points by range threshold.

        Args:
            df: DataFrame with range_m column

        Returns:
            Filtered DataFrame with points within range
        """
        mask = df["range_m"] < self.config.max_range
        return df[mask].copy()

    def statistical_outlier_removal(
        self,
        points: np.ndarray,
        k: int | None = None,
        std_ratio: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove statistical outliers from point cloud.

        Uses the Statistical Outlier Removal (SOR) algorithm:
        1. For each point, compute mean distance to k nearest neighbors
        2. Compute global mean and std of these distances
        3. Remove points where mean distance > global_mean + std_ratio * global_std

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            k: Number of neighbors to consider
            std_ratio: Standard deviation multiplier for threshold

        Returns:
            Tuple of (filtered_points, inlier_mask)
        """
        k = k or self.config.sor_k_neighbors
        std_ratio = std_ratio or self.config.sor_std_ratio

        if len(points) <= k:
            return points, np.ones(len(points), dtype=bool)

        # Build KD-tree for efficient neighbor search
        tree = KDTree(points)

        # Query k+1 neighbors (includes point itself)
        distances, _ = tree.query(points, k=k + 1)

        # Mean distance to k neighbors (excluding self)
        mean_distances = np.mean(distances[:, 1:], axis=1)

        # Global statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)

        # Threshold
        threshold = global_mean + std_ratio * global_std

        # Create inlier mask
        inlier_mask = mean_distances <= threshold

        return points[inlier_mask], inlier_mask

    def preprocess_frame(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Preprocess a single frame of LIDAR data.

        Args:
            df: DataFrame for a single frame

        Returns:
            Tuple of (preprocessed_df, filtered_points)
        """
        # Step 1: Filter by range
        filtered_df = self.filter_by_range(df)

        if len(filtered_df) == 0:
            return filtered_df, np.array([]).reshape(0, 2)

        # Step 2: Extract points
        points = filtered_df[["x", "y"]].values

        return filtered_df, points

        # Step 3: Statistical outlier removal
        filtered_points, inlier_mask = self.statistical_outlier_removal(points)

        # Update DataFrame to match filtered points
        filtered_df = filtered_df.iloc[inlier_mask].copy()

        return filtered_df, filtered_points

    def preprocess(self, df: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
        """Preprocess all frames in the dataset.

        Args:
            df: Full DataFrame with all frames

        Returns:
            Dictionary mapping frame_id to (preprocessed_df, filtered_points)
        """
        results = {}
        frames = sorted(df["frame"].unique())

        for frame_id in frames:
            frame_df = df[df["frame"] == frame_id]
            preprocessed_df, filtered_points = self.preprocess_frame(frame_df)
            results[frame_id] = (preprocessed_df, filtered_points)

        return results
