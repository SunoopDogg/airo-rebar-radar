"""Main execution module for LIDAR-based rebar detection system."""

from pathlib import Path

import json
import numpy as np
import pandas as pd

from .analysis.exporters import create_exporter
from .core.pipeline import ProcessingPipeline, ProcessingResult
from .core.preprocessor import Preprocessor
from .ui.cli import select_csv_file, select_structure_type
from .config.settings import Config
from .config.io_handler import IOHandler
from .ui.roi_selector import ROIBounds, ROISelector
from .ui.structure_adjuster import StructureAdjuster

from .analysis.convergence_analyzer import ConvergenceAnalyzer, ConvergenceThresholds
from .visualization.visualizer import Visualizer


def _select_and_load_file(config: Config) -> tuple[Path, pd.DataFrame, np.ndarray] | None:
    """Select and load a CSV file interactively.

    Args:
        config: Configuration object

    Returns:
        Tuple of (file_path, dataframe, all_points) or None if cancelled
    """
    selected_file = select_csv_file(config.input_dir)
    if not selected_file:
        print("No file selected.")
        return None

    config.structure = select_structure_type()

    print("\nLoading data...")
    df = IOHandler.load_csv(selected_file)
    all_points = df[["x", "y"]].values

    return selected_file, df, all_points


def _configure_roi(
    config: Config,
    all_points: np.ndarray,
    file_name: str,
) -> None:
    """Configure ROI bounds interactively.

    Args:
        config: Configuration object (modified in place)
        all_points: All points from the loaded file
        file_name: Name of the file for display
    """
    default_bounds = config.preprocessing.default_roi_bounds
    current_roi = ROIBounds(
        x_min=config.preprocessing.roi_x_min or -default_bounds,
        x_max=config.preprocessing.roi_x_max or default_bounds,
        y_min=config.preprocessing.roi_y_min or -default_bounds,
        y_max=config.preprocessing.roi_y_max or default_bounds,
    )
    selector = ROISelector()
    selected_roi = selector.select_roi(
        points=all_points,
        title=f"ROI Selection - {file_name}",
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


def _filter_points_by_roi(
    all_points: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Filter points by ROI bounds.

    Args:
        all_points: All points array
        config: Configuration with ROI bounds

    Returns:
        Filtered points within ROI
    """
    preprocessor = Preprocessor(config.preprocessing)
    filtered_points, _ = preprocessor.filter_by_region(all_points)
    return filtered_points


def _adjust_structure_position(
    config: Config,
    roi_points: np.ndarray,
    file_name: str,
) -> None:
    """Adjust structure position interactively.

    Args:
        config: Configuration object (modified in place)
        roi_points: Points within the ROI
        file_name: Name of the file for display
    """
    print("\nAdjusting structure position...")

    adjuster = StructureAdjuster()
    adjusted_position = adjuster.adjust_structure(
        points=roi_points,
        structure=config.structure,
        title=f"Structure Position - {file_name}",
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


def _save_results(
    config: Config,
    result: ProcessingResult,
) -> None:
    """Save detection results using the configured exporter.

    Args:
        config: Configuration object
        result: Processing result containing results
    """
    print("\nSaving results...")

    # Create subdirectory based on file stem
    results_subdir = config.results_dir / result.file_stem
    results_subdir.mkdir(parents=True, exist_ok=True)

    # Create exporter based on config
    exporter = create_exporter(config.export_format)
    ext = "json" if config.export_format.lower() == "json" else "csv"

    # Export detections
    detections_path = results_subdir / f"{result.file_stem}_detections.{ext}"
    if exporter.export_detections(detections_path, result.frame_results):
        print(f"  Saved: {detections_path.relative_to(config.output_dir)}")

    # Export tracks
    tracks_path = results_subdir / f"{result.file_stem}_tracks.{ext}"
    if exporter.export_tracks(tracks_path, result.stable_tracks):
        print(f"  Saved: {tracks_path.relative_to(config.output_dir)}")


def _save_convergence_json(
    summary: dict,
    save_path: Path,
) -> None:
    """Save convergence summary to JSON file.

    Args:
        summary: Summary dictionary from ConvergenceAnalyzer.get_summary()
        save_path: Path to save the JSON file
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _generate_visualizations(
    config: Config,
    result: ProcessingResult,
    all_points: np.ndarray,
    file_name: str,
) -> None:
    """Generate and save visualization plots.

    Args:
        config: Configuration object
        result: Processing result containing results
        all_points: All points from the loaded file
        file_name: Name of the file for display
    """
    print("\nGenerating visualizations...")

    # Create subdirectory based on file stem
    plots_subdir = config.plots_dir / result.file_stem
    plots_subdir.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer(plots_subdir, config.visualization)

    # Raw points plot
    visualizer.plot_raw_points(
        points=all_points,
        save_path=plots_subdir / f"{result.file_stem}_raw_points.png",
    )
    print(
        f"  Saved: {(plots_subdir / f'{result.file_stem}_raw_points.png').relative_to(config.output_dir)}")

    # Per-frame plots (optional)
    if config.visualization.save_frame_plots:
        for frame_result in result.frame_results:
            if frame_result.labels is not None:
                visualizer.plot_with_structure(
                    frame_result.points,
                    frame_result.labels,
                    frame_result.fit_results,
                    config.structure,
                    title=f"Frame {frame_result.frame_id} - Detection",
                    save_path=plots_subdir / f"frame_{frame_result.frame_id:04d}_detection.png",
                )

    # Detection summary plot
    visualizer.plot_detection_summary(
        result.frame_results,
        title=f"Detection Summary - {file_name}",
        save_path=plots_subdir / f"{result.file_stem}_summary.png",
    )
    print(
        f"  Saved: {(plots_subdir / f'{result.file_stem}_summary.png').relative_to(config.output_dir)}")

    # Stable tracks plot
    if result.stable_tracks:
        distance_errors = config.structure.compute_distance_errors(
            result.stable_tracks
        )

        visualizer.plot_tracks(
            result.stable_tracks,
            frame_id=result.n_frames - 1,
            structure=config.structure,
            distance_errors=distance_errors,
            title=f"Stable Tracks - {file_name}",
            save_path=plots_subdir / f"{result.file_stem}_stable_tracks.png",
        )
        print(
            f"  Saved: {(plots_subdir / f'{result.file_stem}_stable_tracks.png').relative_to(config.output_dir)}")

        # Convergence analysis plot
        convergence_analyzer = ConvergenceAnalyzer()
        metrics = convergence_analyzer.analyze_tracks(result.stable_tracks)

        if metrics:
            visualizer.plot_position_convergence(
                metrics,
                title=f"Position Convergence - {file_name}",
                save_path=plots_subdir / f"{result.file_stem}_convergence_position.png",
            )
            visualizer.plot_radius_convergence(
                metrics,
                title=f"Radius Convergence - {file_name}",
                save_path=plots_subdir / f"{result.file_stem}_convergence_radius.png",
            )
            print(
                f"  Saved: {(plots_subdir / f'{result.file_stem}_convergence_position.png').relative_to(config.output_dir)}")
            print(
                f"  Saved: {(plots_subdir / f'{result.file_stem}_convergence_radius.png').relative_to(config.output_dir)}")

            # Convergence summary (console output and JSON)
            thresholds = ConvergenceThresholds(
                position_threshold_mm=config.convergence.position_threshold_mm,
                radius_threshold_mm=config.convergence.radius_threshold_mm,
                consecutive_intervals=config.convergence.consecutive_intervals,
            )
            summary = convergence_analyzer.get_summary(metrics, thresholds)

            # Save JSON
            json_path = plots_subdir / f"{result.file_stem}_convergence_summary.json"
            _save_convergence_json(summary, json_path)
            print(f"  Saved: {json_path.relative_to(config.output_dir)}")


def main():
    """Main entry point."""
    config = Config()

    # Step 1: Select and load file
    load_result = _select_and_load_file(config)
    if load_result is None:
        return

    selected_file, df, all_points = load_result

    # Step 2: Configure ROI
    _configure_roi(config, all_points, selected_file.name)

    # Step 3: Adjust structure position
    roi_points = _filter_points_by_roi(all_points, config)
    _adjust_structure_position(config, roi_points, selected_file.name)

    # Step 4: Process file
    print("=" * 50)
    print(f"Processing: {selected_file.name}")
    pipeline = ProcessingPipeline(config)

    progress_interval = config.processing.progress_log_interval

    def log_progress(current: int, total: int) -> None:
        if current % progress_interval == 0 or current == total:
            print(f"  Processed frame {current}/{total}")

    result = pipeline.process_file(
        selected_file, df=df, progress_callback=log_progress
    )

    print(f"  Total detections: {result.total_detections}")
    print(f"  Stable tracks: {result.n_stable_tracks}")

    # Step 5: Save results
    _save_results(config, result)

    # Step 6: Generate visualizations
    _generate_visualizations(config, result, all_points, selected_file.name)


if __name__ == "__main__":
    main()
