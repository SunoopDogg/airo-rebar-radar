"""File I/O handling for rebar detection system."""

from pathlib import Path

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
        df.columns = df.columns.str.strip()
        return df

