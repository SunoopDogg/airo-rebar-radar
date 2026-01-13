"""Main execution module for LIDAR-based rebar detection system."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import Clusterer
from .circle_fitter import CircleFitter
from .preprocessor import Preprocessor
from .temporal_filter import TemporalFilter, Track
from .utils.cli import select_csv_file
from .utils.config import Config
from .utils.io_handler import IOHandler
from .utils.roi_selector import ROIBounds, ROISelector
from .utils.structure_adjuster import StructureAdjuster
from .visualizer import Visualizer


class RebarDetector:
    """Main class for rebar detection pipeline."""

    def __init__(self, config: Config | None = None):
        """Initialize the rebar detector.

        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.preprocessor = Preprocessor(self.config.preprocessing)
        self.clusterer = Clusterer(self.config.clustering)
        self.circle_fitter = CircleFitter(self.config.circle_fitting)
        self.temporal_filter = TemporalFilter(self.config.kalman_filter)

    def process_frame(self, frame_df: pd.DataFrame, frame_id: int) -> dict:
        """Process a single frame.

        Args:
            frame_df: DataFrame for a single frame
            frame_id: Frame number

        Returns:
            Dictionary with frame processing results and visualization data
        """
        _, points = self.preprocessor.preprocess_frame(frame_df)

        if len(points) == 0:
            return {
                "frame": frame_id,
                "n_detections": 0,
                "detections": [],
                "n_points_raw": len(frame_df),
                "n_points_filtered": 0,
                "points": points,
                "labels": None,
                "fit_results": [],
            }

        labels, clusters = self.clusterer.cluster(points)
        cluster_stats = self.clusterer.get_cluster_stats(labels, points)
        fit_results = self.circle_fitter.fit_clusters(clusters)

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

        if detections and self.config.kalman_filter.enabled:
            detection_tuples = [
                (d["center_x"], d["center_y"], d["radius"])
                for d in detections
            ]
            self.temporal_filter.update(detection_tuples)

        return {
            "frame": frame_id,
            "n_detections": len(detections),
            "detections": detections,
            "n_points_raw": len(frame_df),
            "n_points_filtered": len(points),
            "n_clusters": cluster_stats["n_clusters"],
            "points": points,
            "labels": labels,
            "fit_results": fit_results,
        }

    def _compute_averaged_tracks(
        self, frame_results: list[dict], distance_threshold: float = 0.05
    ) -> list[Track]:
        """Compute averaged tracks from frame detections without Kalman filtering.

        Groups detections across frames by spatial proximity and computes
        average center positions and radii.

        Args:
            frame_results: List of frame processing results
            distance_threshold: Maximum distance to associate detections (meters)

        Returns:
            List of Track objects with averaged values
        """
        # Collect all detections
        all_detections: list[tuple[float, float, float]] = []
        for fr in frame_results:
            for det in fr["detections"]:
                all_detections.append(
                    (det["center_x"], det["center_y"], det["radius"])
                )

        if not all_detections:
            return []

        # Cluster detections by spatial proximity using simple greedy grouping
        groups: list[list[tuple[float, float, float]]] = []

        for det in all_detections:
            det_x, det_y, det_r = det
            assigned = False

            # Try to assign to existing group
            for group in groups:
                # Compute average position of group
                avg_x = np.mean([d[0] for d in group])
                avg_y = np.mean([d[1] for d in group])

                distance = np.sqrt((det_x - avg_x) ** 2 + (det_y - avg_y) ** 2)
                if distance <= distance_threshold:
                    group.append(det)
                    assigned = True
                    break

            # Create new group if not assigned
            if not assigned:
                groups.append([det])

        # Convert groups to Track objects
        tracks: list[Track] = []
        for track_id, group in enumerate(groups):
            if len(group) < 2:  # Require at least 2 detections (like min_hits)
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
    ) -> dict:
        """Process a single CSV file.

        Args:
            file_path: Path to CSV file
            df: Pre-loaded DataFrame (optional, loads from file_path if None)
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            Dictionary with processing results for all frames
        """
        if df is None:
            df = IOHandler.load_csv(file_path)

        n_frames = df["frame"].nunique()
        if self.config.kalman_filter.enabled:
            self.temporal_filter.reset()

        all_results = []
        for frame_id, frame_df in df.groupby("frame", sort=True):
            result = self.process_frame(frame_df, frame_id)
            all_results.append(result)
            if progress_callback:
                progress_callback(len(all_results), n_frames)

        if self.config.kalman_filter.enabled:
            stable_tracks = self.temporal_filter.get_stable_tracks(min_hits=2)
        else:
            stable_tracks = self._compute_averaged_tracks(all_results)

        total_detections = sum(r["n_detections"] for r in all_results)

        return {
            "file": file_path.name,
            "file_stem": file_path.stem,
            "n_frames": n_frames,
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / n_frames if n_frames else 0,
            "n_stable_tracks": len(stable_tracks),
            "stable_tracks": stable_tracks,
            "frame_results": all_results,
        }


def main():
    """Main entry point."""
    config = Config()

    # 1. File selection
    selected_file = select_csv_file(config.input_dir)
    if not selected_file:
        print("No file selected.")
        return

    # 2. Load data
    print("\nLoading data...")
    df = IOHandler.load_csv(selected_file)
    all_points = df[["x", "y"]].values

    # 3. ROI selection
    current_roi = ROIBounds(
        x_min=config.preprocessing.roi_x_min or -1.0,
        x_max=config.preprocessing.roi_x_max or 1.0,
        y_min=config.preprocessing.roi_y_min or -1.0,
        y_max=config.preprocessing.roi_y_max or 1.0,
    )
    selector = ROISelector()
    selected_roi = selector.select_roi(
        points=all_points,
        title=f"ROI Selection - {selected_file.name}",
        default_bounds=current_roi,
    )

    if selected_roi is None:
        print("ROI selection cancelled. Using default ROI.")
    else:
        config.preprocessing.update_roi(
            x_min=selected_roi.x_min,
            x_max=selected_roi.x_max,
            y_min=selected_roi.y_min,
            y_max=selected_roi.y_max,
        )
        print(
            f"ROI updated: X=[{selected_roi.x_min:.3f}, {selected_roi.x_max:.3f}]m, "
            f"Y=[{selected_roi.y_min:.3f}, {selected_roi.y_max:.3f}]m"
        )

    # 4. Structure position adjustment
    print("\nAdjusting structure position...")

    # Filter points by ROI for better context
    roi = config.preprocessing
    roi_mask = (
        (all_points[:, 0] >= (roi.roi_x_min or -float("inf")))
        & (all_points[:, 0] <= (roi.roi_x_max or float("inf")))
        & (all_points[:, 1] >= (roi.roi_y_min or -float("inf")))
        & (all_points[:, 1] <= (roi.roi_y_max or float("inf")))
    )
    roi_points = all_points[roi_mask]

    adjuster = StructureAdjuster()
    adjusted_position = adjuster.adjust_structure(
        points=roi_points,
        structure=config.structure,
        title=f"Structure Position - {selected_file.name}",
    )

    if adjusted_position is None:
        print("Structure adjustment cancelled. Using default position.")
    else:
        config.structure.update_position(
            center_x=adjusted_position.center_x,
            center_y=adjusted_position.center_y,
            yaw=adjusted_position.yaw,
        )
        print(
            f"Structure position updated: "
            f"X={adjusted_position.center_x:.3f}m, "
            f"Y={adjusted_position.center_y:.3f}m, "
            f"Yaw={adjusted_position.yaw_degrees():.1f}deg"
        )

    # 5. Processing
    print("=" * 50)
    print(f"Processing: {selected_file.name}")
    detector = RebarDetector(config)

    def log_progress(current: int, total: int) -> None:
        if current % 10 == 0 or current == total:
            print(f"  Processed frame {current}/{total}")

    summary = detector.process_file(
        selected_file, df=df, progress_callback=log_progress
    )
    n_frames = summary["n_frames"]

    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Stable tracks: {summary['n_stable_tracks']}")

    # 6. Save results
    print("\nSaving results...")
    file_stem = summary["file_stem"]

    if IOHandler.save_detections(
        config.results_dir / f"{file_stem}_detections.csv",
        summary["frame_results"]
    ):
        print(f"  Saved: {file_stem}_detections.csv")

    if IOHandler.save_tracks(
        config.results_dir / f"{file_stem}_tracks.csv",
        summary["stable_tracks"]
    ):
        print(f"  Saved: {file_stem}_tracks.csv")

    # 7. Visualization
    print("\nGenerating visualizations...")
    visualizer = Visualizer(config.plots_dir)

    visualizer.plot_raw_points(
        points=all_points,
        save_path=config.plots_dir / f"{file_stem}_raw_points.png",
    )
    print(f"  Saved: {file_stem}_raw_points.png")

    for fr in summary["frame_results"]:
        if fr["labels"] is not None:
            visualizer.plot_with_structure(
                fr["points"],
                fr["labels"],
                fr["fit_results"],
                config.structure,
                title=f"Frame {fr['frame']} - Detection",
                save_path=config.plots_dir / f"frame_{fr['frame']:04d}_detection.png",
            )

    visualizer.plot_detection_summary(
        summary["frame_results"],
        title=f"Detection Summary - {selected_file.name}",
        save_path=config.plots_dir / f"{file_stem}_summary.png",
    )
    print(f"  Saved: {file_stem}_summary.png")

    if summary["stable_tracks"]:
        distance_errors = config.structure.compute_distance_errors(
            summary["stable_tracks"]
        )

        visualizer.plot_tracks(
            summary["stable_tracks"],
            frame_id=n_frames - 1,
            structure=config.structure,
            distance_errors=distance_errors,
            title=f"Stable Tracks - {selected_file.name}",
            save_path=config.plots_dir / f"{file_stem}_stable_tracks.png",
        )
        print(f"  Saved: {file_stem}_stable_tracks.png")

    # 8. Summary
    print("=" * 50)
    print("Summary:")
    print(f"  Frames: {summary['n_frames']}")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Stable tracks: {summary['n_stable_tracks']}")


if __name__ == "__main__":
    main()
