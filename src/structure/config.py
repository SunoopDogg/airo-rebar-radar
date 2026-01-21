"""Structure configuration for rebar detection system."""

from dataclasses import dataclass
from enum import Enum

from .geometry import calculate_distance
from .position_calculator import TrackPositionCalculator


class Orientation(Enum):
    """Structure orientation."""

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class StructureType(Enum):
    """Structure type for different rebar arrangements."""

    PPVC_LINEAR = "ppvc_linear"  # linear: 4 rebars at 1M spacing
    PPVC_CLUSTER_2 = "ppvc_cluster_2"  # cluster type 2: 2 rebars per cluster
    PPVC_CLUSTER_4 = "ppvc_cluster_4"  # cluster type 4: 4 rebars per cluster
    COLUMN = "column"  # legacy: 2x2 grid column


@dataclass
class StructureConfig:

    # Structure type
    structure_type: StructureType = StructureType.COLUMN

    # Orientation setting
    orientation: Orientation = Orientation.VERTICAL

    # Dimensions in meters (defined as vertical orientation)
    width: float = 0.290        # 290mm (short side)
    height: float = 0.500       # 500mm (long side)

    # Concrete cover in meters
    cover_side: float = 0.080   # 80mm (left/right)
    cover_top: float = 0.090    # 90mm (top)
    cover_bottom: float = 0.090  # 90mm (bottom)

    # Track layout (defined as vertical orientation) - for COLUMN type
    track_count_x: int = 2      # 2 tracks in short-side direction
    track_count_y: int = 2      # 2 tracks in long-side direction
    track_spacing_x: float = 0.130  # 130mm (short-side spacing)
    track_spacing_y: float = 0.320  # 320mm (long-side spacing)

    # Track properties
    track_diameter: float = 0.032   # 32mm (D32)

    # PPVC wall structure parameters
    wall_length: float = 12.0  # Wall length (long side, 12m)
    wall_width: float = 3.4    # Wall width (short side, 3.4m, rebar placement direction)
    cluster_count: int = 12  # Number of clusters along the wall
    cluster_spacing: float = 1.0  # da: 1M spacing between clusters
    cluster_internal_spacing: float = 0.150  # dd: 150mm spacing within cluster
    rebars_per_cluster: int = 1  # 1 for linear, 2 for cluster-2, 4 for cluster-4

    # Structure center position
    center_x: float = -2.1     # X-axis center
    center_y: float = -3.45     # Y-axis center

    # Rotation angle in radians (positive = counter-clockwise)
    yaw: float = 0.0

    def get_display_dimensions(self) -> tuple[float, float]:
        """Get (display_width, display_height) based on structure type and orientation.

        Returns:
            (display_width, display_height): dimensions to apply on X-axis and Y-axis
        """
        # For PPVC types, use actual wall dimensions
        if self.structure_type in (
            StructureType.PPVC_LINEAR,
            StructureType.PPVC_CLUSTER_2,
            StructureType.PPVC_CLUSTER_4,
        ):
            # Use wall_length and wall_width directly
            if self.orientation == Orientation.HORIZONTAL:
                return (self.wall_length, self.wall_width)  # (12m, 3.4m)
            return (self.wall_width, self.wall_length)      # (3.4m, 12m)

        # COLUMN type: original behavior
        if self.orientation == Orientation.HORIZONTAL:
            return (self.height, self.width)  # swap for horizontal
        return (self.width, self.height)  # default: vertical

    def get_display_track_spacing(self) -> tuple[float, float]:
        """Get (spacing_x, spacing_y) based on orientation.

        Returns:
            (spacing_x, spacing_y): spacing to apply on X-axis and Y-axis
        """
        if self.orientation == Orientation.HORIZONTAL:
            return (self.track_spacing_y, self.track_spacing_x)  # swap
        return (self.track_spacing_x, self.track_spacing_y)  # default

    def _get_position_calculator(self) -> TrackPositionCalculator:
        """Create a position calculator for this configuration."""
        return TrackPositionCalculator(
            structure_type=self.structure_type,
            orientation=self.orientation,
            center_x=self.center_x,
            center_y=self.center_y,
            yaw=self.yaw,
        )

    def _get_column_positions(self) -> list[tuple[float, float]]:
        """Calculate track positions for COLUMN type (2x2 grid)."""
        spacing_x, spacing_y = self.get_display_track_spacing()
        calculator = self._get_position_calculator()
        return calculator.calculate_column_positions(
            track_count_x=self.track_count_x,
            track_count_y=self.track_count_y,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
        )

    def _get_linear_positions(self) -> list[tuple[float, float]]:
        """Calculate track positions for PPVC_LINEAR type (single line)."""
        calculator = self._get_position_calculator()
        return calculator.calculate_linear_positions(
            cluster_count=self.cluster_count,
            cluster_spacing=self.cluster_spacing,
            wall_length=self.wall_length,
            wall_thickness=self.width,
        )

    def _get_cluster_positions(self) -> list[tuple[float, float]]:
        """Calculate track positions for PPVC_CLUSTER_2 type."""
        calculator = self._get_position_calculator()
        return calculator.calculate_cluster_positions(
            cluster_count=self.cluster_count,
            cluster_spacing=self.cluster_spacing,
            cluster_internal_spacing=self.cluster_internal_spacing,
            rebars_per_cluster=self.rebars_per_cluster,
            wall_length=self.wall_length,
            wall_thickness=self.width,
        )

    def _get_cluster_4_positions(self) -> list[tuple[float, float]]:
        """Calculate track positions for PPVC_CLUSTER_4 type (2x2 grid per cluster)."""
        calculator = self._get_position_calculator()
        return calculator.calculate_cluster_4_positions(
            cluster_count=self.cluster_count,
            cluster_spacing=self.cluster_spacing,
            cluster_internal_spacing=self.cluster_internal_spacing,
            wall_length=self.wall_length,
            wall_thickness=self.width,
        )

    def get_track_positions(self) -> list[tuple[float, float]]:
        """Calculate expected track positions based on structure type, orientation, and yaw.

        Applies rotation around the structure center (center_x, center_y).

        Returns:
            List of (x, y) tuples for expected rebar positions
        """
        if self.structure_type == StructureType.COLUMN:
            return self._get_column_positions()
        elif self.structure_type == StructureType.PPVC_LINEAR:
            return self._get_linear_positions()
        elif self.structure_type == StructureType.PPVC_CLUSTER_2:
            return self._get_cluster_positions()
        elif self.structure_type == StructureType.PPVC_CLUSTER_4:
            return self._get_cluster_4_positions()
        else:
            # Default to column positions
            return self._get_column_positions()

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
                distance = calculate_distance(
                    track.center_x, track.center_y, exp_x, exp_y
                )
                min_distance = min(min_distance, distance)
            errors[track.track_id] = min_distance

        return errors


def create_ppvc_linear_config(
    center_x: float = 0.0,
    center_y: float = 0.0,
    yaw: float = 0.0,
    orientation: Orientation = Orientation.HORIZONTAL,
) -> StructureConfig:
    return StructureConfig(
        structure_type=StructureType.PPVC_LINEAR,
        orientation=orientation,
        width=0.150,
        wall_length=4.0,
        wall_width=3.4,
        track_diameter=0.025,
        cluster_count=4,
        cluster_spacing=1.0,
        cluster_internal_spacing=0.150,
        rebars_per_cluster=1,
        center_x=center_x,
        center_y=center_y,
        yaw=yaw,
    )


def create_ppvc_cluster_2_config(
    center_x: float = 0.0,
    center_y: float = 0.0,
    yaw: float = 0.0,
    orientation: Orientation = Orientation.HORIZONTAL,
) -> StructureConfig:
    return StructureConfig(
        structure_type=StructureType.PPVC_CLUSTER_2,
        orientation=orientation,
        width=0.150,
        wall_length=4.0,
        wall_width=3.4,
        track_diameter=0.025,
        cluster_count=2,
        cluster_spacing=3.1,
        cluster_internal_spacing=0.150,
        rebars_per_cluster=2,
        center_x=center_x,
        center_y=center_y,
        yaw=yaw,
    )


def create_ppvc_cluster_4_config(
    center_x: float = 0.0,
    center_y: float = 0.0,
    yaw: float = 0.0,
    orientation: Orientation = Orientation.HORIZONTAL,
) -> StructureConfig:
    return StructureConfig(
        structure_type=StructureType.PPVC_CLUSTER_4,
        orientation=orientation,
        width=0.150,
        wall_length=4.0,
        wall_width=3.4,
        track_diameter=0.025,
        cluster_count=2,
        cluster_spacing=3.1,
        cluster_internal_spacing=0.150,
        rebars_per_cluster=4,
        center_x=center_x,
        center_y=center_y,
        yaw=yaw,
    )
