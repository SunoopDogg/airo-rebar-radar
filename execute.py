"""Batch execution script for processing all CSV files in the csv directory."""

from enum import Enum

from src.visualization.visualizer import Visualizer
from src.ui.roi_selector import ROIBounds, ROISelector
from src.structure.config import (
    create_ppvc_linear_config,
    create_ppvc_cluster_2_config,
    create_ppvc_cluster_4_config,
    Orientation,
)
from src.core.preprocessor import Preprocessor
from src.core.pipeline import ProcessingPipeline, ProcessingResult
from src.config.io_handler import IOHandler
from src.config.settings import Config
from src.analysis.convergence_analyzer import ConvergenceAnalyzer, ConvergenceThresholds
from src.analysis.exporters import create_exporter
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============================================================
# Structure Selection Enum
# ============================================================
class StructureSelection(Enum):
    PPVC_LINEAR = "ppvc_linear"
    PPVC_CLUSTER_2 = "ppvc_cluster_2"
    PPVC_CLUSTER_4 = "ppvc_cluster_4"


# ============================================================
# User Configuration Section
# ============================================================
SELECTED_STRUCTURE = StructureSelection.PPVC_CLUSTER_2  # Structure type selection
CSV_SUBFOLDER = "case2"  # Subfolder selection (None = csv root, e.g., "case1")
# ============================================================


# Center X mapping for LINEAR structure
CENTER_X_MAPPING_LINEAR = {
    60: 1.25,
    120: 0.65,
    180: 0.05,
    240: -0.55,
    300: -1.15,
    360: -1.75,
}

# Center X mapping for CLUSTER_2, CLUSTER_4 structures
CENTER_X_MAPPING_CLUSTER = {
    60: 1.175,
    120: 0.575,
    180: -0.025,
    240: -0.625,
    300: -1.225,
    360: -1.825,
}


def get_center_x_from_filename(filename: Path, structure: StructureSelection) -> float:
    """Extract center_x value from filename based on structure type.

    Args:
        filename: Path object of the CSV file (e.g., S3_1-1_60.csv)
        structure: Structure type selection

    Returns:
        center_x value based on the number in filename and structure type
    """
    # Extract the last number from filename stem (e.g., S3_1-1_60 -> 60)
    number = int(filename.stem.split("_")[-1])
    if structure == StructureSelection.PPVC_LINEAR:
        return CENTER_X_MAPPING_LINEAR.get(number, 0.0)
    else:  # CLUSTER_2, CLUSTER_4
        return CENTER_X_MAPPING_CLUSTER.get(number, 0.0)


def configure_roi(
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


def filter_points_by_roi(
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


def save_results(
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


def save_convergence_json(
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


def generate_visualizations(
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
        f"  Saved: {(plots_subdir / f'{result.file_stem}_raw_points.png').relative_to(config.output_dir)}"
    )

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
                    save_path=plots_subdir
                    / f"frame_{frame_result.frame_id:04d}_detection.png",
                )

    # Detection summary plot
    visualizer.plot_detection_summary(
        result.frame_results,
        title=f"Detection Summary - {file_name}",
        save_path=plots_subdir / f"{result.file_stem}_summary.png",
    )
    print(
        f"  Saved: {(plots_subdir / f'{result.file_stem}_summary.png').relative_to(config.output_dir)}"
    )

    # Stable tracks plot
    if result.stable_tracks:
        distance_errors = config.structure.compute_distance_errors(result.stable_tracks)

        visualizer.plot_tracks(
            result.stable_tracks,
            frame_id=result.n_frames - 1,
            structure=config.structure,
            distance_errors=distance_errors,
            title=f"Stable Tracks - {file_name}",
            save_path=plots_subdir / f"{result.file_stem}_stable_tracks.png",
        )
        print(
            f"  Saved: {(plots_subdir / f'{result.file_stem}_stable_tracks.png').relative_to(config.output_dir)}"
        )

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
                f"  Saved: {(plots_subdir / f'{result.file_stem}_convergence_position.png').relative_to(config.output_dir)}"
            )
            print(
                f"  Saved: {(plots_subdir / f'{result.file_stem}_convergence_radius.png').relative_to(config.output_dir)}"
            )

            # Convergence summary (console output and JSON)
            thresholds = ConvergenceThresholds(
                position_threshold_mm=config.convergence.position_threshold_mm,
                radius_threshold_mm=config.convergence.radius_threshold_mm,
                consecutive_intervals=config.convergence.consecutive_intervals,
            )
            summary = convergence_analyzer.get_summary(metrics, thresholds)

            # Save JSON
            json_path = plots_subdir / f"{result.file_stem}_convergence_summary.json"
            save_convergence_json(summary, json_path)
            print(f"  Saved: {json_path.relative_to(config.output_dir)}")


def process_single_file(
    csv_file: Path,
    file_index: int,
    total_files: int,
) -> None:
    """Process a single CSV file.

    Args:
        csv_file: Path to the CSV file
        file_index: Current file index (1-based)
        total_files: Total number of files
    """
    print("\n" + "=" * 60)
    print(f"[{file_index}/{total_files}] Processing: {csv_file.name}")
    print("=" * 60)

    # Create fresh config for each file
    config = Config()

    # Get center_x from filename and create structure based on selection
    center_x = get_center_x_from_filename(csv_file, SELECTED_STRUCTURE)

    if SELECTED_STRUCTURE == StructureSelection.PPVC_LINEAR:
        config.structure = create_ppvc_linear_config(
            center_x=center_x,
            orientation=Orientation.HORIZONTAL,
        )
    elif SELECTED_STRUCTURE == StructureSelection.PPVC_CLUSTER_2:
        config.structure = create_ppvc_cluster_2_config(
            center_x=center_x,
            orientation=Orientation.HORIZONTAL,
        )
    elif SELECTED_STRUCTURE == StructureSelection.PPVC_CLUSTER_4:
        config.structure = create_ppvc_cluster_4_config(
            center_x=center_x,
            orientation=Orientation.HORIZONTAL,
        )
    print(f"Structure: {SELECTED_STRUCTURE.value}, center_x={center_x}")

    # Load file
    print("\nLoading data...")
    df = IOHandler.load_csv(csv_file)
    all_points = df[["x", "y"]].values
    print(f"Loaded {len(all_points)} points")

    # Configure ROI (GUI selection)
    configure_roi(config, all_points, csv_file.name)

    # Process file
    print("\nProcessing...")
    pipeline = ProcessingPipeline(config)

    progress_interval = config.processing.progress_log_interval

    def log_progress(current: int, total: int) -> None:
        if current % progress_interval == 0 or current == total:
            print(f"  Processed frame {current}/{total}")

    result = pipeline.process_file(csv_file, df=df, progress_callback=log_progress)

    print(f"\n  Total detections: {result.total_detections}")
    print(f"  Stable tracks: {result.n_stable_tracks}")

    # Save results
    save_results(config, result)

    # Generate visualizations
    generate_visualizations(config, result, all_points, csv_file.name)

    print(f"\nCompleted: {csv_file.name}")


def main():
    """Main entry point for batch processing."""
    # Get CSV directory with subfolder support
    csv_dir = Path(__file__).parent / "csv"
    if CSV_SUBFOLDER:
        csv_dir = csv_dir / CSV_SUBFOLDER

    if not csv_dir.exists():
        print(f"Error: CSV directory not found: {csv_dir}")
        return

    # Collect all CSV files
    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {csv_dir}")
        return

    print("=" * 60)
    print("Batch Processing - LIDAR Rebar Detection")
    print("=" * 60)
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        center_x = get_center_x_from_filename(f, SELECTED_STRUCTURE)
        print(f"  - {f.name} (center_x={center_x})")

    print(f"\nStructure type: {SELECTED_STRUCTURE.value}")
    print(f"CSV directory: {csv_dir}")
    print("ROI: GUI selection for each file")

    # Process each file
    total_files = len(csv_files)
    for idx, csv_file in enumerate(csv_files, start=1):
        process_single_file(csv_file, idx, total_files)

    print("\n" + "=" * 60)
    print("Batch processing completed!")
    print(f"Processed {total_files} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
