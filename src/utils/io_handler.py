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

    @staticmethod
    def save_detections(
        output_path: Path,
        frame_results: list[dict]
    ) -> bool:
        """Save per-frame detections to CSV.

        Args:
            output_path: Path to save CSV file
            frame_results: List of per-frame result dictionaries

        Returns:
            True if saved, False if no data
        """
        rows = [
            {
                "frame": fr["frame"],
                "center_x": det["center_x"],
                "center_y": det["center_y"],
                "radius": det["radius"],
                "num_points": det["num_points"],
                "fitting_error": det["fitting_error"],
            }
            for fr in frame_results
            for det in fr["detections"]
        ]

        if not rows:
            return False

        pd.DataFrame(rows).to_csv(output_path, index=False)
        return True

    @staticmethod
    def save_tracks(output_path: Path, tracks: list) -> bool:
        """Save stable tracks to CSV.

        Args:
            output_path: Path to save CSV file
            tracks: List of Track objects

        Returns:
            True if saved, False if no data
        """
        if not tracks:
            return False

        rows = [
            {
                "track_id": t.track_id,
                "center_x": t.center_x,
                "center_y": t.center_y,
                "radius": t.radius,
                "hits": t.hits,
                "age": t.age,
            }
            for t in tracks
        ]

        pd.DataFrame(rows).to_csv(output_path, index=False)
        return True

