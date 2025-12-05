"""Main execution module for LIDAR-based rebar detection system."""

import argparse
from pathlib import Path

import pandas as pd

from .clustering import Clusterer
from .circle_fitter import CircleFitter
from .preprocessor import Preprocessor
from .temporal_filter import TemporalFilter
from .utils.config import Config
from .utils.io_handler import IOHandler
from .utils.metrics import Metrics
from .visualizer import Visualizer


class RebarDetector:
    """Main class for rebar detection pipeline."""

    def __init__(self, config: Config | None = None):
        """Initialize the rebar detector.

        Args:
            config: Configuration object
        """
        self.config = config or Config()

        # Initialize modules
        self.preprocessor = Preprocessor(self.config.preprocessing)
        self.clusterer = Clusterer(self.config.clustering)
        self.circle_fitter = CircleFitter(self.config.circle_fitting)
        self.temporal_filter = TemporalFilter(self.config.kalman_filter)
        self.visualizer = Visualizer(self.config.plots_dir)
        self.io_handler = IOHandler()

    def process_frame(
        self,
        frame_df: pd.DataFrame,
        frame_id: int,
        visualize: bool = False
    ) -> dict:
        """Process a single frame.

        Args:
            frame_df: DataFrame for a single frame
            frame_id: Frame number
            visualize: Whether to generate visualizations

        Returns:
            Dictionary with frame processing results
        """
        # Step 1: Preprocess
        preprocessed_df, points = self.preprocessor.preprocess_frame(frame_df)

        if len(points) == 0:
            return {
                "frame": frame_id,
                "n_detections": 0,
                "detections": [],
                "n_points_raw": len(frame_df),
                "n_points_filtered": 0,
            }

        # Step 2: Clustering
        labels, clusters = self.clusterer.cluster(points)
        cluster_stats = self.clusterer.get_cluster_stats(labels, points)

        # Step 3: Circle fitting
        fit_results = self.circle_fitter.fit_clusters(clusters)

        # Step 4: Calculate metrics and prepare detections
        detections = []
        for i, result in enumerate(fit_results):
            if result.success:
                confidence = Metrics.calculate_confidence(
                    result.num_points,
                    result.residual
                )
                radius_conf = Metrics.calculate_radius_confidence(
                    result.radius,
                    self.config.circle_fitting.min_radius,
                    self.config.circle_fitting.max_radius
                )
                combined_conf = 0.7 * confidence + 0.3 * radius_conf

                detections.append({
                    "center_x": result.center_x,
                    "center_y": result.center_y,
                    "radius": result.radius,
                    "confidence": combined_conf,
                    "num_points": result.num_points,
                    "fitting_error": result.residual,
                })

        # Step 5: Temporal filtering
        if detections:
            detection_tuples = [
                (d["center_x"], d["center_y"], d["radius"], d["confidence"])
                for d in detections
            ]
            self.temporal_filter.update(detection_tuples)

        # Visualize if requested
        if visualize and len(points) > 0:
            save_path = self.config.plots_dir / f"frame_{frame_id:04d}_clusters.png"
            self.visualizer.plot_circle_fits(
                points, labels, fit_results,
                title=f"Frame {frame_id} - Circle Fitting",
                save_path=save_path
            )

        avg_conf = sum(d["confidence"] for d in detections) / len(detections) if detections else 0

        return {
            "frame": frame_id,
            "n_detections": len(detections),
            "detections": detections,
            "n_points_raw": len(frame_df),
            "n_points_filtered": len(points),
            "n_clusters": cluster_stats["n_clusters"],
            "avg_confidence": avg_conf,
        }

    def process_file(
        self,
        file_path: Path,
        visualize: bool = True,
        save_results: bool = True
    ) -> dict:
        """Process a single CSV file.

        Args:
            file_path: Path to CSV file
            visualize: Whether to generate visualizations
            save_results: Whether to save results to file

        Returns:
            Dictionary with processing results for all frames
        """
        print(f"Processing: {file_path.name}")

        # Load data
        df = self.io_handler.load_csv(file_path)
        frames = self.io_handler.get_frames(df)

        print(f"  Found {len(frames)} frames")

        # Reset temporal filter for new file
        self.temporal_filter.reset()

        # Process each frame
        all_results = []
        for frame_id in frames:
            frame_df = self.io_handler.get_frame_data(df, frame_id)
            result = self.process_frame(frame_df, frame_id, visualize=visualize)
            all_results.append(result)

            # Print progress
            if (frame_id + 1) % 10 == 0 or frame_id == frames[-1]:
                print(f"  Processed frame {frame_id + 1}/{len(frames)}")

        # Get stable tracks
        stable_tracks = self.temporal_filter.get_stable_tracks(min_hits=2)

        # Summary statistics
        total_detections = sum(r["n_detections"] for r in all_results)
        avg_detections = total_detections / len(all_results) if all_results else 0

        summary = {
            "file": file_path.name,
            "n_frames": len(frames),
            "total_detections": total_detections,
            "avg_detections_per_frame": avg_detections,
            "n_stable_tracks": len(stable_tracks),
            "frame_results": all_results,
        }

        # Save results
        if save_results:
            self._save_results(file_path.stem, all_results, stable_tracks)

        # Generate summary visualization
        if visualize:
            save_path = self.config.plots_dir / f"{file_path.stem}_summary.png"
            self.visualizer.plot_detection_summary(
                all_results,
                title=f"Detection Summary - {file_path.name}",
                save_path=save_path
            )

        print(f"  Total detections: {total_detections}")
        print(f"  Stable tracks: {len(stable_tracks)}")

        return summary

    def _save_results(
        self,
        file_stem: str,
        frame_results: list[dict],
        stable_tracks: list
    ) -> None:
        """Save processing results to files.

        Args:
            file_stem: Base name for output files
            frame_results: List of per-frame results
            stable_tracks: List of stable Track objects
        """
        # Save per-frame detections
        rows = []
        for fr in frame_results:
            for det in fr["detections"]:
                rows.append({
                    "frame": fr["frame"],
                    "center_x": det["center_x"],
                    "center_y": det["center_y"],
                    "radius": det["radius"],
                    "confidence": det["confidence"],
                    "num_points": det["num_points"],
                    "fitting_error": det["fitting_error"],
                })

        if rows:
            df = pd.DataFrame(rows)
            output_path = self.config.results_dir / f"{file_stem}_detections.csv"
            self.io_handler.save_results(df, output_path)
            print(f"  Saved detections to: {output_path}")

        # Save stable tracks
        if stable_tracks:
            track_rows = []
            for track in stable_tracks:
                track_rows.append({
                    "track_id": track.track_id,
                    "center_x": track.center_x,
                    "center_y": track.center_y,
                    "radius": track.radius,
                    "confidence": track.confidence,
                    "hits": track.hits,
                    "age": track.age,
                })
            df = pd.DataFrame(track_rows)
            output_path = self.config.results_dir / f"{file_stem}_tracks.csv"
            self.io_handler.save_results(df, output_path)
            print(f"  Saved tracks to: {output_path}")

    def process_all(
        self,
        visualize: bool = True,
        save_results: bool = True
    ) -> list[dict]:
        """Process all CSV files in the input directory.

        Args:
            visualize: Whether to generate visualizations
            save_results: Whether to save results to files

        Returns:
            List of summaries for all processed files
        """
        csv_files = sorted(self.config.input_dir.glob("*.csv"))

        if not csv_files:
            print(f"No CSV files found in {self.config.input_dir}")
            return []

        print(f"Found {len(csv_files)} CSV files")
        print("=" * 50)

        summaries = []
        for csv_file in csv_files:
            summary = self.process_file(csv_file, visualize, save_results)
            summaries.append(summary)
            print("-" * 50)

        # Print overall summary
        print("=" * 50)
        print("Overall Summary:")
        total_frames = sum(s["n_frames"] for s in summaries)
        total_detections = sum(s["total_detections"] for s in summaries)
        total_tracks = sum(s["n_stable_tracks"] for s in summaries)
        print(f"  Total files processed: {len(summaries)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Total detections: {total_detections}")
        print(f"  Total stable tracks: {total_tracks}")

        return summaries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LIDAR-based rebar center axis detection system"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=Path("csv"),
        help="Directory containing CSV files (default: csv)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization generation"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving results to files"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="DBSCAN eps parameter (default: 0.05)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="DBSCAN min_samples parameter (default: 3)"
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=2.0,
        help="Maximum range filter in meters (default: 2.0)"
    )

    args = parser.parse_args()

    # Create configuration
    config = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        results_dir=args.output_dir / "results",
        plots_dir=args.output_dir / "plots",
    )
    config.preprocessing.max_range = args.max_range
    config.clustering.eps = args.eps
    config.clustering.min_samples = args.min_samples

    # Run detection
    detector = RebarDetector(config)
    detector.process_all(
        visualize=not args.no_visualize,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
