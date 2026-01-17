"""Results export functionality with multiple format support."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..core.pipeline import FrameResult
from ..core.temporal_filter import Track
from ..config.logging import get_logger

logger = get_logger(__name__)


class ResultsExporter(ABC):
    """Abstract base class for results exporters.

    Defines the interface for exporting detection results to various formats.
    """

    @abstractmethod
    def export_detections(
        self,
        output_path: Path,
        frame_results: list[FrameResult],
    ) -> bool:
        """Export per-frame detections.

        Args:
            output_path: Path to save the output file
            frame_results: List of frame results to export

        Returns:
            True if export succeeded, False otherwise
        """
        pass

    @abstractmethod
    def export_tracks(
        self,
        output_path: Path,
        tracks: list[Track],
    ) -> bool:
        """Export stable tracks.

        Args:
            output_path: Path to save the output file
            tracks: List of Track objects to export

        Returns:
            True if export succeeded, False otherwise
        """
        pass


class CSVExporter(ResultsExporter):
    """Export results to CSV format."""

    def export_detections(
        self,
        output_path: Path,
        frame_results: list[FrameResult],
    ) -> bool:
        """Export per-frame detections to CSV.

        Args:
            output_path: Path to save the CSV file
            frame_results: List of frame results

        Returns:
            True if export succeeded, False otherwise
        """
        rows = []
        for fr in frame_results:
            for det in fr.detections:
                rows.append({
                    "frame": fr.frame_id,
                    "center_x": det["center_x"],
                    "center_y": det["center_y"],
                    "radius": det["radius"],
                    "num_points": det["num_points"],
                    "fitting_error": det["fitting_error"],
                })

        if not rows:
            logger.warning("No detections to export to %s", output_path)
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Exported %d detections to %s", len(rows), output_path)
        return True

    def export_tracks(
        self,
        output_path: Path,
        tracks: list[Track],
    ) -> bool:
        """Export stable tracks to CSV.

        Args:
            output_path: Path to save the CSV file
            tracks: List of Track objects

        Returns:
            True if export succeeded, False otherwise
        """
        if not tracks:
            logger.warning("No tracks to export to %s", output_path)
            return False

        rows = []
        for t in tracks:
            row = {
                "track_id": t.track_id,
                "center_x": t.center_x,
                "center_y": t.center_y,
                "radius": t.radius,
                "hits": t.hits,
                "age": t.age,
            }
            for interval_idx, avg in t.interval_averages.items():
                row[f"{interval_idx}0_center_x"] = avg["center_x"]
                row[f"{interval_idx}0_center_y"] = avg["center_y"]
                row[f"{interval_idx}0_radius"] = avg["radius"]
            rows.append(row)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Exported %d tracks to %s", len(rows), output_path)
        return True


class JSONExporter(ResultsExporter):
    """Export results to JSON format."""

    def __init__(self, indent: int = 2):
        """Initialize JSON exporter.

        Args:
            indent: Indentation level for JSON output
        """
        self.indent = indent

    def export_detections(
        self,
        output_path: Path,
        frame_results: list[FrameResult],
    ) -> bool:
        """Export per-frame detections to JSON.

        Args:
            output_path: Path to save the JSON file
            frame_results: List of frame results

        Returns:
            True if export succeeded, False otherwise
        """
        data = [
            {
                "frame": fr.frame_id,
                "n_detections": fr.n_detections,
                "detections": fr.detections,
            }
            for fr in frame_results
        ]

        if not data or all(len(d["detections"]) == 0 for d in data):
            logger.warning("No detections to export to %s", output_path)
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"frames": data}, f, indent=self.indent)

        total_detections = sum(len(d["detections"]) for d in data)
        logger.info("Exported %d detections to %s", total_detections, output_path)
        return True

    def export_tracks(
        self,
        output_path: Path,
        tracks: list[Track],
    ) -> bool:
        """Export stable tracks to JSON.

        Args:
            output_path: Path to save the JSON file
            tracks: List of Track objects

        Returns:
            True if export succeeded, False otherwise
        """
        if not tracks:
            logger.warning("No tracks to export to %s", output_path)
            return False

        data = [
            {
                "track_id": t.track_id,
                "center_x": t.center_x,
                "center_y": t.center_y,
                "radius": t.radius,
                "hits": t.hits,
                "age": t.age,
                "interval_averages": t.interval_averages,
            }
            for t in tracks
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"tracks": data}, f, indent=self.indent)

        logger.info("Exported %d tracks to %s", len(data), output_path)
        return True


def create_exporter(format: str = "csv") -> ResultsExporter:
    """Factory function to create an exporter for the specified format.

    Args:
        format: Export format ("csv" or "json")

    Returns:
        ResultsExporter instance

    Raises:
        ValueError: If format is not supported
    """
    format_lower = format.lower()
    if format_lower == "csv":
        return CSVExporter()
    elif format_lower == "json":
        return JSONExporter()
    else:
        raise ValueError(f"Unsupported export format: {format}. Use 'csv' or 'json'.")
