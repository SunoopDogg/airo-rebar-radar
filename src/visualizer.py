"""Visualization module for rebar detection results."""

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
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

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
                # Noise points
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

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

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

        # Plot noise points
        noise_mask = labels == -1
        if np.any(noise_mask):
            ax.scatter(
                points[noise_mask, 0], points[noise_mask, 1],
                s=5, c="gray", alpha=0.3, label="Noise"
            )

        # Plot each cluster and its fitted circle
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            color = self.colors[i % len(self.colors)]

            # Plot cluster points
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

                # Mark center
                ax.plot(
                    result.center_x, result.center_y,
                    "x", color=color, markersize=10, markeredgewidth=1
                )

                # Add label
                ax.annotate(
                    f"r={result.radius*1000:.1f}mm",
                    (result.center_x, result.center_y),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=8,
                    color=color
                )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_tracks(
        self,
        tracks: list[Track],
        frame_id: int,
        points: np.ndarray | None = None,
        structure: StructureConfig | None = None,
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
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw structure overlay if provided
        if structure is not None:
            self.draw_structure_overlay(ax, structure)

        # Plot raw points if provided
        if points is not None and len(points) > 0:
            ax.scatter(
                points[:, 0], points[:, 1],
                s=5, c="lightgray", alpha=0.5, label="Points"
            )

        # Plot each track
        for i, track in enumerate(tracks):
            color = self.colors[track.track_id % len(self.colors)]

            # Plot circle
            circle = plt.Circle(
                (track.center_x, track.center_y),
                track.radius,
                fill=False,
                color=color,
                linewidth=2
            )
            ax.add_patch(circle)

            # Mark center
            ax.plot(
                track.center_x, track.center_y,
                "o", color=color, markersize=8
            )

            # Add track info
            ax.annotate(
                f"ID:{track.track_id}\nhits:{track.hits}",
                (track.center_x, track.center_y),
                textcoords="offset points",
                xytext=(15, 15),
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )

        title = title or f"Tracked Rebars - Frame {frame_id}"
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Set axis limits based on structure if provided
        if structure is not None:
            margin = 0.1
            ax.set_xlim(
                structure.center_x - structure.width / 2 - margin,
                structure.center_x + structure.width / 2 + margin
            )
            ax.set_ylim(
                structure.center_y - structure.height / 2 - margin,
                structure.center_y + structure.height / 2 + margin
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def draw_structure_overlay(
        self,
        ax: plt.Axes,
        structure: StructureConfig
    ) -> None:
        """Draw structure overlay on existing axes.

        Args:
            ax: Matplotlib axes to draw on
            structure: Structure configuration with dimensions
        """
        # Calculate structure boundary corners
        half_width = structure.width / 2
        half_height = structure.height / 2

        left = structure.center_x - half_width
        right = structure.center_x + half_width
        bottom = structure.center_y - half_height
        top = structure.center_y + half_height

        # Draw concrete outline (outer rectangle)
        rect = mpatches.Rectangle(
            (left, bottom),
            structure.width,
            structure.height,
            fill=False,
            edgecolor="gray",
            linewidth=2,
            linestyle="-",
            label="Concrete outline"
        )
        ax.add_patch(rect)

        # Draw expected rebar positions
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

        # Draw structure overlay first (background)
        self.draw_structure_overlay(ax, structure)

        unique_labels = set(labels)
        unique_labels.discard(-1)

        # Plot noise points
        noise_mask = labels == -1
        if np.any(noise_mask):
            ax.scatter(
                points[noise_mask, 0], points[noise_mask, 1],
                s=5, c="gray", alpha=0.3, label="Noise"
            )

        # Plot each cluster and its fitted circle
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            color = self.colors[i % len(self.colors)]

            # Plot cluster points
            ax.scatter(
                points[mask, 0], points[mask, 1],
                s=10, c=[color], alpha=0.8
            )

            if i < len(fit_results):
                result = fit_results[i]
                # Detected rebar circle (solid red)
                circle = plt.Circle(
                    (result.center_x, result.center_y),
                    result.radius,
                    fill=False,
                    color="red",
                    linewidth=2,
                    linestyle="-"
                )
                ax.add_patch(circle)

                # Mark center
                ax.plot(
                    result.center_x, result.center_y,
                    "x", color="red", markersize=10, markeredgewidth=2
                )

                # Add label
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

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color="gray", linestyle="-",
                       linewidth=2, label="Concrete outline"),
            plt.Line2D([0], [0], color="blue", linestyle=":",
                       linewidth=1.5, label="Expected rebar"),
            plt.Line2D([0], [0], color="red", linestyle="-", linewidth=2, label="Detected rebar"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Set axis limits with margin
        margin = 0.1
        ax.set_xlim(
            structure.center_x - structure.width/2 - margin,
            structure.center_x + structure.width/2 + margin
        )
        ax.set_ylim(
            structure.center_y - structure.height/2 - margin,
            structure.center_y + structure.height/2 + margin
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

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

        # Number of detections per frame
        axes[0].plot(frames, n_detections, "b-o", markersize=4)
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Number of Detections")
        axes[0].set_title("Detections per Frame")
        axes[0].grid(True, alpha=0.3)

        # Detection positions scatter
        all_x = []
        all_y = []
        for d in all_detections:
            if "detections" in d:
                for det in d["detections"]:
                    all_x.append(det["center_x"])
                    all_y.append(det["center_y"])

        if all_x:
            axes[1].scatter(all_x, all_y, s=10, alpha=0.5)
            axes[1].set_xlabel("X (m)")
            axes[1].set_ylabel("Y (m)")
            axes[1].set_title("All Detection Positions")
            axes[1].set_aspect("equal")
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
