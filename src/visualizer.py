"""Visualization module for rebar detection results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .circle_fitter import CircleFitResult
from .temporal_filter import Track


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
                s=20, c=[color], alpha=0.8
            )

            # Plot fitted circle if successful
            if i < len(fit_results) and fit_results[i].success:
                result = fit_results[i]
                circle = plt.Circle(
                    (result.center_x, result.center_y),
                    result.radius,
                    fill=False,
                    color=color,
                    linewidth=2,
                    linestyle="--"
                )
                ax.add_patch(circle)

                # Mark center
                ax.plot(
                    result.center_x, result.center_y,
                    "x", color=color, markersize=10, markeredgewidth=2
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
        title: str | None = None,
        save_path: Path | None = None,
        show: bool = False
    ) -> None:
        """Plot tracked rebars.

        Args:
            tracks: List of Track objects
            frame_id: Current frame number
            points: Optional raw points to show
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))

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
                f"ID:{track.track_id}\nconf:{track.confidence:.2f}",
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        frames = [d["frame"] for d in all_detections]
        n_detections = [d["n_detections"] for d in all_detections]

        # Number of detections per frame
        axes[0, 0].plot(frames, n_detections, "b-o", markersize=4)
        axes[0, 0].set_xlabel("Frame")
        axes[0, 0].set_ylabel("Number of Detections")
        axes[0, 0].set_title("Detections per Frame")
        axes[0, 0].grid(True, alpha=0.3)

        # Average confidence per frame
        avg_conf = [d.get("avg_confidence", 0) for d in all_detections]
        axes[0, 1].plot(frames, avg_conf, "g-o", markersize=4)
        axes[0, 1].set_xlabel("Frame")
        axes[0, 1].set_ylabel("Average Confidence")
        axes[0, 1].set_title("Confidence per Frame")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        # Detection positions scatter
        all_x = []
        all_y = []
        for d in all_detections:
            if "detections" in d:
                for det in d["detections"]:
                    all_x.append(det["center_x"])
                    all_y.append(det["center_y"])

        if all_x:
            axes[1, 0].scatter(all_x, all_y, s=10, alpha=0.5)
            axes[1, 0].set_xlabel("X (m)")
            axes[1, 0].set_ylabel("Y (m)")
            axes[1, 0].set_title("All Detection Positions")
            axes[1, 0].set_aspect("equal")
            axes[1, 0].grid(True, alpha=0.3)

        # Radius distribution
        all_radii = []
        for d in all_detections:
            if "detections" in d:
                for det in d["detections"]:
                    all_radii.append(det["radius"] * 1000)  # Convert to mm

        if all_radii:
            axes[1, 1].hist(all_radii, bins=20, edgecolor="black", alpha=0.7)
            axes[1, 1].set_xlabel("Radius (mm)")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Radius Distribution")
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
