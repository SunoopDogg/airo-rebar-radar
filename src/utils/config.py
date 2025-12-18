"""Configuration management for rebar detection system."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    max_range: float = 2.0  # Maximum range in meters
    sor_k_neighbors: int = 10  # Number of neighbors for SOR
    sor_std_ratio: float = 1.0  # Standard deviation ratio for SOR
    roi_x_min: float | None = 1.0  # Region of Interest x minimum
    roi_x_max: float | None = 1.5  # Region of Interest x maximum
    roi_y_min: float | None = -1.0  # Region of Interest y minimum
    roi_y_max: float | None = 0.0  # Region of Interest y maximum


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
    process_noise: float = 0.001  # Process noise covariance
    measurement_noise: float = 0.01  # Measurement noise covariance
    max_distance: float = 0.1  # Maximum distance for track association


@dataclass
class StructureConfig:
    """Structure geometry configuration for visualization overlay.

    Vertical column cross-section:
    - width: X-axis direction (290mm) - short side
    - height: Y-axis direction (500mm) - long side (vertical)
    """
    # Dimensions in meters (vertical orientation)
    width: float = 0.290        # 290mm (X-axis, short side)
    height: float = 0.500       # 500mm (Y-axis, long side)

    # Concrete cover in meters
    cover_side: float = 0.080   # 80mm (left/right)
    cover_top: float = 0.090    # 90mm (top)
    cover_bottom: float = 0.090  # 90mm (bottom)

    # Track layout (vertical orientation)
    track_count_x: int = 2      # 2 tracks in X-axis
    track_count_y: int = 2      # 2 tracks in Y-axis
    track_spacing_x: float = 0.130  # 130mm (X-axis spacing)
    track_spacing_y: float = 0.320  # 320mm (Y-axis spacing)

    # Track properties
    track_diameter: float = 0.032   # 32mm (D32)

    # Structure center position
    center_x: float = 1.25      # 1.25m from sensor
    center_y: float = -0.5      # Y-axis center

    def get_track_positions(self) -> list[tuple[float, float]]:
        """Calculate expected track positions based on structure geometry."""
        positions = []

        for i in range(self.track_count_x):
            for j in range(self.track_count_y):
                x = self.center_x + \
                    (i - 0.5) * self.track_spacing_x if self.track_count_x == 2 else self.center_x
                y = self.center_y + \
                    (j - 0.5) * self.track_spacing_y if self.track_count_y == 2 else self.center_y
                positions.append((x, y))

        return positions


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
