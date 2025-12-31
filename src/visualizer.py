"""Visualization module for rebar detection results."""

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .circle_fitter import CircleFitResult
from .temporal_filter import Track
from .utils.config import StructureConfig


class Visualizer:
    """Visualize LIDAR data and detection results."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir or Path("output/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("default")
        self.colors = plt.cm.tab10.colors

    def _setup_axes(self, ax: plt.Axes, title: str) -> None:
        """Apply common axes settings."""
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    def _finalize_plot(
        self, fig: plt.Figure, save_path: Path | None, show: bool
    ) -> None:
        """Save, show, and close the figure."""
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_noise_points(
        self, ax: plt.Axes, points: np.ndarray, labels: np.ndarray
    ) -> None:
        """Plot noise points (label == -1) in gray."""
        noise_mask = labels == -1
        if np.any(noise_mask):
            ax.scatter(
                points[noise_mask, 0], points[noise_mask, 1],
                s=5, c="gray", alpha=0.3, label="Noise"
            )

    def _set_structure_limits(
        self, ax: plt.Axes, structure: StructureConfig, margin: float = 0.1
    ) -> None:
        """Set axis limits based on structure dimensions and orientation."""
        display_width, display_height = structure.get_display_dimensions()

        ax.set_xlim(
            structure.center_x - display_width / 2 - margin,
            structure.center_x + display_width / 2 + margin
        )
        ax.set_ylim(
            structure.center_y - display_height / 2 - margin,
            structure.center_y + display_height / 2 + margin
        )

    def plot_raw_points(
        self,
        points: np.ndarray,
        title: str = "Raw LIDAR Points",
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot raw LIDAR points.

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6, c="blue")
        self._setup_axes(ax, title)
        self._finalize_plot(fig, save_path, show)

    def plot_clusters(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        title: str = "Clustered Points",
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot clustered points with different colors.

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            labels: Cluster labels (-1 for noise)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = set(labels)

        for label in unique_labels:
            mask = labels == label
            if label == -1:
                ax.scatter(
                    points[mask, 0], points[mask, 1],
                    s=5, c="gray", alpha=0.3, label="Noise"
                )
            else:
                color = self.colors[label % len(self.colors)]
                ax.scatter(
                    points[mask, 0], points[mask, 1],
                    s=20, c=[color], alpha=0.8, label=f"Cluster {label}"
                )

        self._setup_axes(ax, title)
        ax.legend(loc="upper right")
        self._finalize_plot(fig, save_path, show)

    def plot_circle_fits(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        fit_results: list[CircleFitResult],
        title: str = "Circle Fitting Results",
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot points with fitted circles.

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            labels: Cluster labels
            fit_results: List of circle fit results
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        unique_labels = set(labels)
        unique_labels.discard(-1)

        self._plot_noise_points(ax, points, labels)

        # Plot each cluster and its fitted circle
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            color = self.colors[i % len(self.colors)]

            ax.scatter(
                points[mask, 0], points[mask, 1],
                s=5, c=[color], alpha=0.8
            )

            if i < len(fit_results):
                result = fit_results[i]
                circle = plt.Circle(
                    (result.center_x, result.center_y),
                    result.radius,
                    fill=False,
                    color=color,
                    linewidth=1,
                    linestyle="--"
                )
                ax.add_patch(circle)

                ax.plot(
                    result.center_x, result.center_y,
                    "x", color=color, markersize=10, markeredgewidth=1
                )

                ax.annotate(
                    f"r={result.radius*1000:.1f}mm",
                    (result.center_x, result.center_y),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=8,
                    color=color
                )

        self._setup_axes(ax, title)
        self._finalize_plot(fig, save_path, show)

    def plot_tracks(
        self,
        tracks: list[Track],
        frame_id: int,
        points: np.ndarray | None = None,
        structure: StructureConfig | None = None,
        distance_errors: dict[int, float] | None = None,
        title: str | None = None,
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot tracked rebars.

        Args:
            tracks: List of Track objects
            frame_id: Current frame number
            points: Optional raw points to show
            structure: Optional structure config for overlay
            distance_errors: Optional dictionary mapping track_id to distance error in meters
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        if structure is not None:
            self.draw_structure_overlay(ax, structure)

        if points is not None and len(points) > 0:
            ax.scatter(
                points[:, 0], points[:, 1],
                s=5, c="lightgray", alpha=0.5, label="Points"
            )

        for track in tracks:
            color = self.colors[track.track_id % len(self.colors)]

            circle = plt.Circle(
                (track.center_x, track.center_y),
                track.radius,
                fill=False,
                color=color,
                linewidth=2
            )
            ax.add_patch(circle)

            ax.plot(
                track.center_x, track.center_y,
                "o", color=color, markersize=8
            )

            annotation_lines = [f"ID:{track.track_id}", f"hits:{track.hits}"]
            if distance_errors is not None and track.track_id in distance_errors:
                error_mm = distance_errors[track.track_id] * 1000
                annotation_lines.append(f"err:{error_mm:.2f}mm")

            ax.annotate(
                "\n".join(annotation_lines),
                (track.center_x, track.center_y),
                textcoords="offset points",
                xytext=(15, 15),
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )

        self._setup_axes(ax, title or f"Tracked Rebars - Frame {frame_id}")

        if structure is not None:
            self._set_structure_limits(ax, structure)

        self._finalize_plot(fig, save_path, show)

    def draw_structure_overlay(
        self,
        ax: plt.Axes,
        structure: StructureConfig
    ) -> None:
        """Draw structure overlay on existing axes with rotation support.

        Args:
            ax: Matplotlib axes to draw on
            structure: Structure configuration with dimensions and yaw
        """
        # Get display dimensions based on orientation
        display_width, display_height = structure.get_display_dimensions()

        # Calculate structure boundary corners
        half_width = display_width / 2
        half_height = display_height / 2

        # Define corners relative to center (before rotation)
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]

        # Apply rotation and translation
        cos_yaw = math.cos(structure.yaw)
        sin_yaw = math.sin(structure.yaw)
        rotated_corners = []
        for dx, dy in corners:
            x = dx * cos_yaw - dy * sin_yaw + structure.center_x
            y = dx * sin_yaw + dy * cos_yaw + structure.center_y
            rotated_corners.append((x, y))

        # Draw concrete outline as polygon (supports rotation)
        poly = mpatches.Polygon(
            rotated_corners,
            fill=False,
            edgecolor="gray",
            linewidth=2,
            linestyle="-",
            label="Concrete outline"
        )
        ax.add_patch(poly)

        # Draw expected rebar positions (already includes rotation via get_track_positions)
        track_positions = structure.get_track_positions()
        track_radius = structure.track_diameter / 2

        for x, y in track_positions:
            # Expected rebar circle (blue, dashed)
            circle = plt.Circle(
                (x, y),
                track_radius,
                fill=False,
                edgecolor="blue",
                linewidth=1.5,
                linestyle=":",
                alpha=0.7
            )
            ax.add_patch(circle)

            # Center marker
            ax.plot(x, y, "+", color="blue", markersize=8, markeredgewidth=1.5, alpha=0.7)

    def plot_with_structure(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        fit_results: list[CircleFitResult],
        structure: StructureConfig,
        title: str = "Detection with Structure Overlay",
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot detection results with structure overlay.

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            labels: Cluster labels
            fit_results: List of circle fit results
            structure: Structure configuration
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        self.draw_structure_overlay(ax, structure)

        unique_labels = set(labels)
        unique_labels.discard(-1)

        self._plot_noise_points(ax, points, labels)

        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            color = self.colors[i % len(self.colors)]

            ax.scatter(
                points[mask, 0], points[mask, 1],
                s=10, c=[color], alpha=0.8
            )

            if i < len(fit_results):
                result = fit_results[i]
                circle = plt.Circle(
                    (result.center_x, result.center_y),
                    result.radius,
                    fill=False,
                    color="red",
                    linewidth=2,
                    linestyle="-"
                )
                ax.add_patch(circle)

                ax.plot(
                    result.center_x, result.center_y,
                    "x", color="red", markersize=10, markeredgewidth=2
                )

                ax.annotate(
                    f"r={result.radius*1000:.1f}mm",
                    (result.center_x, result.center_y),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=9,
                    color="red",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
                )

        legend_elements = [
            plt.Line2D([0], [0], color="gray", linestyle="-",
                       linewidth=2, label="Concrete outline"),
            plt.Line2D([0], [0], color="blue", linestyle=":",
                       linewidth=1.5, label="Expected rebar"),
            plt.Line2D([0], [0], color="red", linestyle="-",
                       linewidth=2, label="Detected rebar"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        self._setup_axes(ax, title)
        self._set_structure_limits(ax, structure)
        self._finalize_plot(fig, save_path, show)

    def plot_detection_summary(
        self,
        all_detections: list[dict],
        title: str = "Detection Summary",
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot summary of all detections across frames.

        Args:
            all_detections: List of detection dictionaries per frame
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        frames = [d["frame"] for d in all_detections]
        n_detections = [d["n_detections"] for d in all_detections]

        axes[0].plot(frames, n_detections, "b-o", markersize=4)
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Number of Detections")
        axes[0].set_title("Detections per Frame")
        axes[0].grid(True, alpha=0.3)

        all_x = []
        all_y = []
        for d in all_detections:
            for det in d.get("detections", []):
                all_x.append(det["center_x"])
                all_y.append(det["center_y"])

        if all_x:
            axes[1].scatter(all_x, all_y, s=10, alpha=0.5)
            self._setup_axes(axes[1], "All Detection Positions")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        self._finalize_plot(fig, save_path, show)
