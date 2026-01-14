"""Visualization module for rebar detection results."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .circle_fitter import CircleFitResult
from .structure_renderer import StructureRenderer, create_structure_legend_elements
from .temporal_filter import Track
from .utils.config import VisualizationConfig
from .utils.structure import StructureConfig

if TYPE_CHECKING:
    from .pipeline import FrameResult


class Visualizer:
    """Visualize LIDAR data and detection results."""

    def __init__(
        self,
        output_dir: Path | None = None,
        config: VisualizationConfig | None = None
    ):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
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
            plt.savefig(save_path, dpi=self.config.default_dpi, bbox_inches="tight")
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
        self, ax: plt.Axes, structure: StructureConfig, margin: float | None = None
    ) -> None:
        """Set axis limits based on structure dimensions and orientation.

        Args:
            ax: Matplotlib axes to set limits on
            structure: Structure configuration
            margin: Optional margin to add around structure.
                    If None, uses 10% of the larger dimension or 0.1m minimum.
        """
        renderer = StructureRenderer(structure)
        renderer.set_axis_limits(ax, margin)

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
        fig, ax = plt.subplots(figsize=self.config.figure_size_standard)

        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6, c="blue")
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
        fig, ax = plt.subplots(figsize=self.config.figure_size_large)

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
                error_mm = distance_errors[track.track_id] * self.config.mm_per_meter
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
        renderer = StructureRenderer(structure)
        renderer.draw_full_overlay(ax)

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
        fig, ax = plt.subplots(figsize=self.config.figure_size_large)

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
                    f"r={result.radius * self.config.mm_per_meter:.1f}mm",
                    (result.center_x, result.center_y),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=9,
                    color="red",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
                )

        legend_elements = create_structure_legend_elements()
        ax.legend(handles=legend_elements, loc="upper right")

        self._setup_axes(ax, title)
        self._set_structure_limits(ax, structure)
        self._finalize_plot(fig, save_path, show)

    def plot_detection_summary(
        self,
        all_detections: list[dict] | list["FrameResult"],
        title: str = "Detection Summary",
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot summary of all detections across frames.

        Args:
            all_detections: List of detection dictionaries or FrameResult objects per frame
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size_summary)

        # Extract frame IDs and detection counts, handling both dict and FrameResult
        frames = []
        n_detections = []
        for d in all_detections:
            if hasattr(d, "frame_id"):
                # FrameResult dataclass
                frames.append(d.frame_id)
                n_detections.append(d.n_detections)
            else:
                # dict
                frames.append(d["frame"])
                n_detections.append(d["n_detections"])

        axes[0].plot(frames, n_detections, "b-o", markersize=4)
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Number of Detections")
        axes[0].set_title("Detections per Frame")
        axes[0].grid(True, alpha=0.3)

        all_x = []
        all_y = []
        for d in all_detections:
            # Handle both dict and FrameResult
            if hasattr(d, "detections"):
                detections = d.detections
            else:
                detections = d.get("detections", [])

            for det in detections:
                all_x.append(det["center_x"])
                all_y.append(det["center_y"])

        if all_x:
            axes[1].scatter(all_x, all_y, s=10, alpha=0.5)
            self._setup_axes(axes[1], "All Detection Positions")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        self._finalize_plot(fig, save_path, show)
