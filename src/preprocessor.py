"""Preprocessing module for LIDAR point cloud data."""

import numpy as np
import pandas as pd

from .utils.config import PreprocessingConfig


class Preprocessor:
    """Preprocess LIDAR point cloud data."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def filter_by_region(
        self,
        points: np.ndarray,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter points within rectangular region.

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            x_min: Minimum x coordinate (default from config)
            x_max: Maximum x coordinate (default from config)
            y_min: Minimum y coordinate (default from config)
            y_max: Maximum y coordinate (default from config)

        Returns:
            Tuple of (filtered_points, inlier_mask)
        """
        if len(points) == 0:
            return points, np.array([], dtype=bool)

        # Use config values if not provided
        x_min = x_min if x_min is not None else self.config.roi_x_min
        x_max = x_max if x_max is not None else self.config.roi_x_max
        y_min = y_min if y_min is not None else self.config.roi_y_min
        y_max = y_max if y_max is not None else self.config.roi_y_max

        # Start with all True mask
        mask = np.ones(len(points), dtype=bool)

        # Apply filters for each boundary (skip if None)
        if x_min is not None:
            mask &= points[:, 0] >= x_min
        if x_max is not None:
            mask &= points[:, 0] <= x_max
        if y_min is not None:
            mask &= points[:, 1] >= y_min
        if y_max is not None:
            mask &= points[:, 1] <= y_max

        return points[mask], mask

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
        if len(df) == 0:
            return df, np.array([]).reshape(0, 2)

        # Step 1: Extract points
        points = df[["x", "y"]].values

        # Step 2: Filter by region
        filtered_points, region_mask = self.filter_by_region(points)

        return df[region_mask], filtered_points
