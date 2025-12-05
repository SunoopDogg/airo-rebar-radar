"""Evaluation metrics for rebar detection system."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionMetrics:
    """Metrics for a single detection."""
    center_x: float
    center_y: float
    radius: float
    confidence: float
    num_points: int
    fitting_error: float


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_id: int
    num_detections: int
    detections: list[DetectionMetrics]


class Metrics:
    """Calculate and store evaluation metrics."""

    @staticmethod
    def calculate_fitting_error(
        points: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float
    ) -> float:
        """Calculate mean squared error of circle fit.

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            center_x: Circle center x coordinate
            center_y: Circle center y coordinate
            radius: Circle radius

        Returns:
            Mean squared error of distances from points to circle
        """
        if len(points) == 0:
            return float("inf")

        distances = np.sqrt(
            (points[:, 0] - center_x) ** 2 +
            (points[:, 1] - center_y) ** 2
        )
        errors = distances - radius
        return float(np.mean(errors ** 2))

    @staticmethod
    def calculate_confidence(
        num_points: int,
        fitting_error: float,
        min_points: int = 3,
        error_threshold: float = 0.001
    ) -> float:
        """Calculate detection confidence score.

        Confidence is based on:
        - Number of points (more points = higher confidence)
        - Fitting error (lower error = higher confidence)

        Args:
            num_points: Number of points in cluster
            fitting_error: Circle fitting MSE
            min_points: Minimum expected points
            error_threshold: Error threshold for normalization

        Returns:
            Confidence score between 0 and 1
        """
        # Point count factor (saturates at 10 points)
        point_factor = min(1.0, (num_points - min_points + 1) / 7)

        # Error factor (exponential decay)
        error_factor = np.exp(-fitting_error / error_threshold)

        # Combined confidence
        confidence = 0.5 * point_factor + 0.5 * error_factor
        return float(np.clip(confidence, 0, 1))

    @staticmethod
    def calculate_radius_confidence(
        radius: float,
        expected_min: float = 0.005,
        expected_max: float = 0.05
    ) -> float:
        """Calculate confidence based on radius being in expected range.

        Args:
            radius: Estimated radius
            expected_min: Minimum expected radius
            expected_max: Maximum expected radius

        Returns:
            Confidence score between 0 and 1
        """
        if expected_min <= radius <= expected_max:
            return 1.0
        elif radius < expected_min:
            return max(0, radius / expected_min)
        else:
            return max(0, 1 - (radius - expected_max) / expected_max)

    @staticmethod
    def aggregate_frame_metrics(
        frame_metrics_list: list[FrameMetrics]
    ) -> dict:
        """Aggregate metrics across all frames.

        Args:
            frame_metrics_list: List of per-frame metrics

        Returns:
            Dictionary with aggregated statistics
        """
        if not frame_metrics_list:
            return {
                "total_frames": 0,
                "total_detections": 0,
                "avg_detections_per_frame": 0,
                "avg_confidence": 0,
                "avg_radius": 0,
            }

        all_detections = []
        for fm in frame_metrics_list:
            all_detections.extend(fm.detections)

        total_detections = len(all_detections)

        return {
            "total_frames": len(frame_metrics_list),
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / len(frame_metrics_list),
            "avg_confidence": (
                np.mean([d.confidence for d in all_detections])
                if all_detections else 0
            ),
            "avg_radius": (
                np.mean([d.radius for d in all_detections])
                if all_detections else 0
            ),
        }
