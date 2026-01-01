"""Configuration management for rebar detection system."""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Orientation(Enum):
    """Structure orientation."""

    VERTICAL = "vertical"  # 세로 방향 (height > width)
    HORIZONTAL = "horizontal"  # 가로 방향 (width > height)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    roi_x_min: float | None = -1.0  # Region of Interest x minimum
    roi_x_max: float | None = 1.0  # Region of Interest x maximum
    roi_y_min: float | None = -1.0  # Region of Interest y minimum
    roi_y_max: float | None = 1.0  # Region of Interest y maximum

    lidar_offset_x: float = 0.0  # X offset in meters
    lidar_offset_y: float = 0.0   # Y offset in meters

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
    # max_radius: float = 0.0168  # Maximum expected rebar radius (meters) - 32mm +5%
    # min_radius: float = 0.0152  # Minimum expected rebar radius (meters) - 32mm -5%
    max_radius: float = 0.0176  # Maximum expected rebar radius (meters)
    min_radius: float = 0.0144  # Minimum expected rebar radius (meters)


@dataclass
class KalmanFilterConfig:
    """Kalman filter configuration for temporal stabilization."""
    enabled: bool = False  # Enable/disable Kalman filtering
    process_noise: float = 0.001  # Process noise covariance
    measurement_noise: float = 0.01  # Measurement noise covariance
    max_distance: float = 0.1  # Maximum distance for track association


@dataclass
class StructureConfig:
    """Structure geometry configuration for visualization overlay.

    Column cross-section dimensions (defined in vertical orientation):
    - width: short side (290mm)
    - height: long side (500mm)

    Use orientation field to switch between vertical/horizontal display.
    """

    # Orientation setting
    orientation: Orientation = Orientation.VERTICAL

    # Dimensions in meters (defined as vertical orientation)
    width: float = 0.290        # 290mm (short side)
    height: float = 0.500       # 500mm (long side)

    # Concrete cover in meters
    cover_side: float = 0.080   # 80mm (left/right)
    cover_top: float = 0.090    # 90mm (top)
    cover_bottom: float = 0.090  # 90mm (bottom)

    # Track layout (defined as vertical orientation)
    track_count_x: int = 2      # 2 tracks in short-side direction
    track_count_y: int = 2      # 2 tracks in long-side direction
    track_spacing_x: float = 0.130  # 130mm (short-side spacing)
    track_spacing_y: float = 0.320  # 320mm (long-side spacing)

    # Track properties
    track_diameter: float = 0.032   # 32mm (D32)

    # Structure center position
    center_x: float = -2.1     # X-axis center
    center_y: float = -3.45     # Y-axis center

    # Rotation angle in radians (positive = counter-clockwise)
    yaw: float = 0.0

    def get_display_dimensions(self) -> tuple[float, float]:
        """Get (display_width, display_height) based on orientation.

        Returns:
            (display_width, display_height): X축, Y축에 적용할 치수
        """
        if self.orientation == Orientation.HORIZONTAL:
            return (self.height, self.width)  # swap for horizontal
        return (self.width, self.height)  # default: vertical

    def get_display_track_spacing(self) -> tuple[float, float]:
        """Get (spacing_x, spacing_y) based on orientation.

        Returns:
            (spacing_x, spacing_y): X축, Y축에 적용할 간격
        """
        if self.orientation == Orientation.HORIZONTAL:
            return (self.track_spacing_y, self.track_spacing_x)  # swap
        return (self.track_spacing_x, self.track_spacing_y)  # default

    def get_track_positions(self) -> list[tuple[float, float]]:
        """Calculate expected track positions based on structure geometry, orientation, and yaw.

        Applies rotation around the structure center (center_x, center_y).
        """
        positions = []
        spacing_x, spacing_y = self.get_display_track_spacing()
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        for i in range(self.track_count_x):
            for j in range(self.track_count_y):
                # Calculate position relative to center (before rotation)
                dx = (i - 0.5) * spacing_x if self.track_count_x == 2 else 0.0
                dy = (j - 0.5) * spacing_y if self.track_count_y == 2 else 0.0

                # Apply rotation around center
                x = self.center_x + dx * cos_yaw - dy * sin_yaw
                y = self.center_y + dx * sin_yaw + dy * cos_yaw
                positions.append((x, y))

        return positions

    def update_position(
        self, center_x: float, center_y: float, yaw: float
    ) -> None:
        """Update structure position from interactive selection.

        Args:
            center_x: Structure center X coordinate
            center_y: Structure center Y coordinate
            yaw: Rotation angle in radians
        """
        self.center_x = center_x
        self.center_y = center_y
        self.yaw = yaw

    def compute_distance_errors(self, tracks: list) -> dict[int, float]:
        """Compute distance errors from track centers to nearest expected rebar.

        Args:
            tracks: List of Track objects (track_id, center_x, center_y)

        Returns:
            Dictionary mapping track_id to distance error in meters
        """
        expected_positions = self.get_track_positions()
        errors = {}

        for track in tracks:
            min_distance = float('inf')
            for exp_x, exp_y in expected_positions:
                distance = math.sqrt(
                    (track.center_x - exp_x) ** 2 +
                    (track.center_y - exp_y) ** 2
                )
                min_distance = min(min_distance, distance)
            errors[track.track_id] = min_distance

        return errors


@dataclass
class Config:
    """Main configuration class."""
    # Paths
    input_dir: Path = field(default_factory=lambda: Path("csv"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    results_dir: Path = field(default_factory=lambda: Path("output/results"))
    plots_dir: Path = field(default_factory=lambda: Path("output/plots"))

    # Module configs
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    circle_fitting: CircleFittingConfig = field(default_factory=CircleFittingConfig)
    kalman_filter: KalmanFilterConfig = field(default_factory=KalmanFilterConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)

    def __post_init__(self):
        """Ensure output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
