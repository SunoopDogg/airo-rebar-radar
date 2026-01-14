"""Preprocessing module for LIDAR point cloud data."""

import numpy as np
import pandas as pd

from .utils.config import PreprocessingConfig
from .utils.logging import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Preprocess LIDAR point cloud data."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def _apply_lidar_offset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply LIDAR origin offset to transform coordinates.

        Transforms from sensor-local frame to reference frame:
            x_new = x + lidar_offset_x
            y_new = y + lidar_offset_y

        Args:
            df: DataFrame with 'x' and 'y' columns

        Returns:
            DataFrame with transformed coordinates
        """
        if self.config.lidar_offset_x == 0.0 and self.config.lidar_offset_y == 0.0:
            return df
        df = df.copy()
        df["x"] = df["x"] + self.config.lidar_offset_x
        df["y"] = df["y"] + self.config.lidar_offset_y
        return df

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
            logger.debug("Empty DataFrame received for preprocessing")
            return df, np.array([]).reshape(0, 2)

        # Validate required columns
        required_columns = {"x", "y"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.error("Missing required columns: %s", missing_columns)
            return df, np.array([]).reshape(0, 2)

        df = self._apply_lidar_offset(df)
        points = df[["x", "y"]].values
        filtered_points, region_mask = self.filter_by_region(points)

        if len(filtered_points) == 0:
            logger.debug("No points remain after ROI filtering")

        return df[region_mask], filtered_points
