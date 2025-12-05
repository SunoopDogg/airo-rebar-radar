"""File I/O handling for rebar detection system."""

from pathlib import Path

import numpy as np
import pandas as pd


class IOHandler:
    """Handle file input/output operations."""

    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        """Load LIDAR CSV data.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with columns: frame, idx, angle_rad, range_m, x, y
        """
        df = pd.read_csv(file_path)
        # Normalize column names (remove spaces)
        df.columns = df.columns.str.strip()
        return df

    @staticmethod
    def load_all_csvs(input_dir: Path) -> dict[str, pd.DataFrame]:
        """Load all CSV files from directory.

        Args:
            input_dir: Directory containing CSV files

        Returns:
            Dictionary mapping filename to DataFrame
        """
        data = {}
        for csv_file in sorted(input_dir.glob("*.csv")):
            data[csv_file.stem] = IOHandler.load_csv(csv_file)
        return data

    @staticmethod
    def save_results(
        results: pd.DataFrame,
        output_path: Path,
        include_header: bool = True
    ) -> None:
        """Save detection results to CSV.

        Args:
            results: DataFrame with detection results
            output_path: Output file path
            include_header: Whether to include column headers
        """
        results.to_csv(output_path, index=False, header=include_header)

    @staticmethod
    def extract_points(df: pd.DataFrame) -> np.ndarray:
        """Extract x, y points from DataFrame.

        Args:
            df: DataFrame with x, y columns

        Returns:
            Numpy array of shape (N, 2) with x, y coordinates
        """
        return df[["x", "y"]].values

    @staticmethod
    def get_frames(df: pd.DataFrame) -> list[int]:
        """Get unique frame numbers from DataFrame.

        Args:
            df: DataFrame with frame column

        Returns:
            Sorted list of unique frame numbers
        """
        return sorted(df["frame"].unique())

    @staticmethod
    def get_frame_data(df: pd.DataFrame, frame: int) -> pd.DataFrame:
        """Get data for a specific frame.

        Args:
            df: Full DataFrame
            frame: Frame number

        Returns:
            DataFrame filtered to specific frame
        """
        return df[df["frame"] == frame].copy()
