"""Configuration management for rebar detection system."""

from dataclasses import dataclass, field
from pathlib import Path

from .structure import StructureConfig


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    roi_x_min: float | None = -1.0
    roi_x_max: float | None = 1.0
    roi_y_min: float | None = -1.0
    roi_y_max: float | None = 1.0

    lidar_offset_x: float = 0.0
    lidar_offset_y: float = 0.0

    # Default ROI bounds when no bounds are specified
    default_roi_bounds: float = 1.0

    def update_roi(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        """Update ROI bounds from interactive selection.

        Args:
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate
            y_min: Minimum y coordinate
            y_max: Maximum y coordinate
        """
        self.roi_x_min = x_min
        self.roi_x_max = x_max
        self.roi_y_min = y_min
        self.roi_y_max = y_max


@dataclass
class ClusteringConfig:
    """DBSCAN clustering configuration."""
    eps: float = 0.05  # Maximum distance between points in a cluster (meters)
    min_samples: int = 3  # Minimum points to form a cluster


@dataclass
class CircleFittingConfig:
    """Circle fitting configuration."""
    min_points: int = 3  # Minimum points required for circle fitting
    max_radius: float = 0.0176  # Maximum expected rebar radius (meters)
    min_radius: float = 0.0144  # Minimum expected rebar radius (meters)
    collinearity_tolerance: float = 1e-10  # Tolerance for collinearity check


@dataclass
class KalmanFilterConfig:
    """Kalman filter configuration for temporal stabilization."""
    enabled: bool = False  # Enable/disable Kalman filtering
    process_noise: float = 0.001  # Process noise covariance
    measurement_noise: float = 0.01  # Measurement noise covariance
    max_distance: float = 0.1  # Maximum distance for track association
    max_consecutive_misses: int = 3  # Max frames without detection before track deletion
    time_step: float = 1.0  # Time step for Kalman filter state transition
    radius_process_noise_scale: float = 0.1  # Scaling factor for radius process noise
    radius_measurement_noise_scale: float = 0.5  # Scaling factor for radius measurement noise


@dataclass
class TrackingConfig:
    """Configuration for track aggregation and averaging."""
    min_track_detections: int = 2  # Minimum detections required for stable track
    distance_threshold: float = 0.05  # Maximum distance to associate detections (meters)


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    default_dpi: int = 150  # Default DPI for saved plots
    figure_size_standard: tuple[int, int] = (10, 8)  # Standard figure size
    figure_size_large: tuple[int, int] = (12, 10)  # Large figure size
    figure_size_summary: tuple[int, int] = (14, 5)  # Summary figure size
    mm_per_meter: int = 1000  # Conversion factor for display
    save_frame_plots: bool = True  # Whether to save individual frame plots


@dataclass
class ProcessingConfig:
    """Configuration for processing behavior."""
    progress_log_interval: int = 10  # Log progress every N frames


@dataclass
class Config:
    """Main configuration class."""
    # Paths
    input_dir: Path = field(default_factory=lambda: Path("csv"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    results_dir: Path = field(default_factory=lambda: Path("output/results"))
    plots_dir: Path = field(default_factory=lambda: Path("output/plots"))

    # Export settings
    export_format: str = "csv"  # Export format: "csv" or "json"

    # Module configs
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    circle_fitting: CircleFittingConfig = field(default_factory=CircleFittingConfig)
    kalman_filter: KalmanFilterConfig = field(default_factory=KalmanFilterConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)

    def __post_init__(self):
        """Ensure output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
