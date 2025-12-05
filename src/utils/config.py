"""Configuration management for rebar detection system."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    max_range: float = 2.0  # Maximum range in meters
    sor_k_neighbors: int = 10  # Number of neighbors for SOR
    sor_std_ratio: float = 1.0  # Standard deviation ratio for SOR


@dataclass
class ClusteringConfig:
    """DBSCAN clustering configuration."""
    eps: float = 0.05  # Maximum distance between points in a cluster (meters)
    min_samples: int = 3  # Minimum points to form a cluster


@dataclass
class CircleFittingConfig:
    """Circle fitting configuration."""
    min_points: int = 3  # Minimum points required for circle fitting
    max_radius: float = 0.05  # Maximum expected rebar radius (meters)
    min_radius: float = 0.005  # Minimum expected rebar radius (meters)


@dataclass
class KalmanFilterConfig:
    """Kalman filter configuration for temporal stabilization."""
    process_noise: float = 0.001  # Process noise covariance
    measurement_noise: float = 0.01  # Measurement noise covariance
    max_distance: float = 0.1  # Maximum distance for track association


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

    def __post_init__(self):
        """Ensure output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        preprocessing = PreprocessingConfig(**config_dict.get("preprocessing", {}))
        clustering = ClusteringConfig(**config_dict.get("clustering", {}))
        circle_fitting = CircleFittingConfig(**config_dict.get("circle_fitting", {}))
        kalman_filter = KalmanFilterConfig(**config_dict.get("kalman_filter", {}))

        return cls(
            input_dir=Path(config_dict.get("input_dir", "csv")),
            output_dir=Path(config_dict.get("output_dir", "output")),
            results_dir=Path(config_dict.get("results_dir", "output/results")),
            plots_dir=Path(config_dict.get("plots_dir", "output/plots")),
            preprocessing=preprocessing,
            clustering=clustering,
            circle_fitting=circle_fitting,
            kalman_filter=kalman_filter,
        )
