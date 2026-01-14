"""Processing pipeline abstraction for rebar detection."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .circle_fitter import CircleFitResult, CircleFitter
from .clustering import Clusterer
from .preprocessor import Preprocessor
from .temporal_filter import TemporalFilter, Track
from .utils.config import Config
from .utils.io_handler import IOHandler


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    frame_id: int
    n_detections: int
    detections: list[dict]
    n_points_raw: int
    n_points_filtered: int
    n_clusters: int
    points: np.ndarray
    labels: np.ndarray | None
    fit_results: list[CircleFitResult]


@dataclass
class ProcessingResult:
    """Result of processing a complete file."""

    file_name: str
    file_stem: str
    n_frames: int
    total_detections: int
    avg_detections_per_frame: float
    n_stable_tracks: int
    stable_tracks: list[Track]
    frame_results: list[FrameResult]


class ProcessingPipeline:
    """Configurable processing pipeline for rebar detection.

    This class orchestrates the processing stages and provides a clean
    interface for running the detection pipeline.
    """

    def __init__(self, config: Config):
        """Initialize the processing pipeline.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize components
        self.preprocessor = Preprocessor(config.preprocessing)
        self.clusterer = Clusterer(config.clustering)
        self.circle_fitter = CircleFitter(config.circle_fitting)
        self.temporal_filter = TemporalFilter(config.kalman_filter)

    def process_frame(self, frame_df: pd.DataFrame, frame_id: int) -> FrameResult:
        """Process a single frame through all pipeline stages.

        Args:
            frame_df: DataFrame for a single frame
            frame_id: Frame number

        Returns:
            FrameResult with all processing results
        """
        # Stage 1: Preprocessing
        _, points = self.preprocessor.preprocess_frame(frame_df)

        if len(points) == 0:
            return FrameResult(
                frame_id=frame_id,
                n_detections=0,
                detections=[],
                n_points_raw=len(frame_df),
                n_points_filtered=0,
                n_clusters=0,
                points=points,
                labels=None,
                fit_results=[],
            )

        # Stage 2: Clustering
        labels, clusters = self.clusterer.cluster(points)
        cluster_stats = self.clusterer.get_cluster_stats(labels, points)

        # Stage 3: Circle Fitting
        fit_results = self.circle_fitter.fit_clusters(clusters)

        # Convert to detection dictionaries
        detections = [
            {
                "center_x": r.center_x,
                "center_y": r.center_y,
                "radius": r.radius,
                "num_points": r.num_points,
                "fitting_error": r.residual,
            }
            for r in fit_results
        ]

        # Optional: Update temporal filter
        if detections and self.config.kalman_filter.enabled:
            detection_tuples = [
                (d["center_x"], d["center_y"], d["radius"])
                for d in detections
            ]
            self.temporal_filter.update(detection_tuples)

        return FrameResult(
            frame_id=frame_id,
            n_detections=len(detections),
            detections=detections,
            n_points_raw=len(frame_df),
            n_points_filtered=len(points),
            n_clusters=cluster_stats["n_clusters"],
            points=points,
            labels=labels,
            fit_results=fit_results,
        )

    def _compute_averaged_tracks(
        self, frame_results: list[FrameResult]
    ) -> list[Track]:
        """Compute averaged tracks from frame detections.

        Args:
            frame_results: List of frame processing results

        Returns:
            List of Track objects with averaged values
        """
        distance_threshold = self.config.tracking.distance_threshold
        min_detections = self.config.tracking.min_track_detections

        all_detections: list[tuple[float, float, float]] = []
        for frame_result in frame_results:
            for det in frame_result.detections:
                all_detections.append(
                    (det["center_x"], det["center_y"], det["radius"])
                )

        if not all_detections:
            return []

        groups: list[list[tuple[float, float, float]]] = []

        for det in all_detections:
            det_x, det_y, det_r = det
            assigned = False

            for group in groups:
                avg_x = np.mean([d[0] for d in group])
                avg_y = np.mean([d[1] for d in group])

                distance = np.sqrt((det_x - avg_x) ** 2 + (det_y - avg_y) ** 2)
                if distance <= distance_threshold:
                    group.append(det)
                    assigned = True
                    break

            if not assigned:
                groups.append([det])

        tracks: list[Track] = []
        for track_id, group in enumerate(groups):
            if len(group) < min_detections:
                continue

            avg_x = float(np.mean([d[0] for d in group]))
            avg_y = float(np.mean([d[1] for d in group]))
            avg_r = float(np.mean([d[2] for d in group]))

            track = Track(
                track_id=track_id,
                center_x=avg_x,
                center_y=avg_y,
                radius=avg_r,
                hits=len(group),
            )
            tracks.append(track)

        return tracks

    def process_file(
        self,
        file_path: Path,
        df: pd.DataFrame | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ProcessingResult:
        """Process a complete CSV file through the pipeline.

        Args:
            file_path: Path to CSV file
            df: Pre-loaded DataFrame (optional, loads from file_path if None)
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            ProcessingResult with all results
        """
        if df is None:
            df = IOHandler.load_csv(file_path)

        n_frames = df["frame"].nunique()
        if self.config.kalman_filter.enabled:
            self.temporal_filter.reset()

        all_results: list[FrameResult] = []
        for frame_id, frame_df in df.groupby("frame", sort=True):
            result = self.process_frame(frame_df, frame_id)
            all_results.append(result)
            if progress_callback:
                progress_callback(len(all_results), n_frames)

        # Compute stable tracks
        if self.config.kalman_filter.enabled:
            stable_tracks = self.temporal_filter.get_stable_tracks(min_hits=2)
        else:
            stable_tracks = self._compute_averaged_tracks(all_results)

        total_detections = sum(r.n_detections for r in all_results)

        return ProcessingResult(
            file_name=file_path.name,
            file_stem=file_path.stem,
            n_frames=n_frames,
            total_detections=total_detections,
            avg_detections_per_frame=total_detections / n_frames if n_frames else 0,
            n_stable_tracks=len(stable_tracks),
            stable_tracks=stable_tracks,
            frame_results=all_results,
        )
