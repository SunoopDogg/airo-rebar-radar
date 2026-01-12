"""Track position calculation for different structure types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .geometry import rotate_and_translate

if TYPE_CHECKING:
    from .structure import Orientation, StructureType


class TrackPositionCalculator:
    """Calculate track positions for different structure types.

    This class handles the coordinate calculation for rebar positions
    based on structure type, orientation, and position parameters.
    """

    def __init__(
        self,
        structure_type: StructureType,
        orientation: Orientation,
        center_x: float,
        center_y: float,
        yaw: float,
    ):
        """Initialize the calculator.

        Args:
            structure_type: Type of structure (COLUMN, PPVC_LINEAR, etc.)
            orientation: Structure orientation (VERTICAL or HORIZONTAL)
            center_x: X coordinate of structure center
            center_y: Y coordinate of structure center
            yaw: Rotation angle in radians
        """
        self.structure_type = structure_type
        self.orientation = orientation
        self.center_x = center_x
        self.center_y = center_y
        self.yaw = yaw

    def calculate_column_positions(
        self,
        track_count_x: int,
        track_count_y: int,
        spacing_x: float,
        spacing_y: float,
    ) -> list[tuple[float, float]]:
        """Calculate track positions for COLUMN type (2x2 grid).

        Args:
            track_count_x: Number of tracks in X direction
            track_count_y: Number of tracks in Y direction
            spacing_x: Spacing between tracks in X direction (meters)
            spacing_y: Spacing between tracks in Y direction (meters)

        Returns:
            List of (x, y) tuples for track positions
        """
        positions = []

        for i in range(track_count_x):
            for j in range(track_count_y):
                # Calculate position relative to center (before rotation)
                dx = (i - 0.5) * spacing_x if track_count_x == 2 else 0.0
                dy = (j - 0.5) * spacing_y if track_count_y == 2 else 0.0

                # Apply rotation around center
                x, y = rotate_and_translate(
                    dx, dy, self.center_x, self.center_y, self.yaw
                )
                positions.append((x, y))

        return positions

    def calculate_linear_positions(
        self,
        cluster_count: int,
        cluster_spacing: float,
        wall_length: float,
        wall_thickness: float,
    ) -> list[tuple[float, float]]:
        """Calculate track positions for PPVC_LINEAR type (single line).

        Rebars are placed along the wall_width direction,
        offset from the left edge by wall thickness.

        Args:
            cluster_count: Number of rebar positions
            cluster_spacing: Spacing between rebars (meters)
            wall_length: Length of the wall (meters)
            wall_thickness: Wall thickness for offset (meters)

        Returns:
            List of (x, y) tuples for track positions
        """
        from .structure import Orientation

        positions = []

        # Rebars are placed along wall_width direction
        total_span = (cluster_count - 1) * cluster_spacing
        start_offset = -total_span / 2

        # Offset from left edge of wall by wall thickness
        wall_edge_offset = -wall_length / 2 + wall_thickness

        for i in range(cluster_count):
            # Position along wall_width direction (before rotation)
            if self.orientation == Orientation.HORIZONTAL:
                dx = wall_edge_offset
                dy = start_offset + i * cluster_spacing
            else:
                dx = start_offset + i * cluster_spacing
                dy = wall_edge_offset

            # Apply rotation around center
            x, y = rotate_and_translate(
                dx, dy, self.center_x, self.center_y, self.yaw
            )
            positions.append((x, y))

        return positions

    def calculate_cluster_positions(
        self,
        cluster_count: int,
        cluster_spacing: float,
        cluster_internal_spacing: float,
        rebars_per_cluster: int,
        wall_length: float,
        wall_thickness: float,
    ) -> list[tuple[float, float]]:
        """Calculate track positions for PPVC_CLUSTER_2 type.

        Clusters are placed along wall_width direction.
        Rebars within each cluster are placed along wall_length direction.

        Args:
            cluster_count: Number of clusters
            cluster_spacing: Spacing between clusters (meters)
            cluster_internal_spacing: Spacing within cluster (meters)
            rebars_per_cluster: Number of rebars per cluster
            wall_length: Length of the wall (meters)
            wall_thickness: Wall thickness for offset (meters)

        Returns:
            List of (x, y) tuples for track positions
        """
        from .structure import Orientation

        positions = []

        # Offset from left edge of wall by wall thickness
        wall_edge_offset = -wall_length / 2 + wall_thickness

        # Clusters are placed along wall_width direction (centered)
        total_span = (cluster_count - 1) * cluster_spacing
        start_offset = -total_span / 2

        for i in range(cluster_count):
            # Cluster center position along wall_width direction
            cluster_center = start_offset + i * cluster_spacing

            # Generate positions for each rebar in the cluster
            for j in range(rebars_per_cluster):
                # Internal offset within cluster (along wall_length direction)
                internal_offset = (
                    (j - (rebars_per_cluster - 1) / 2)
                    * cluster_internal_spacing
                )

                # Position before rotation
                if self.orientation == Orientation.HORIZONTAL:
                    dx = wall_edge_offset + internal_offset
                    dy = cluster_center
                else:
                    dx = cluster_center
                    dy = wall_edge_offset + internal_offset

                # Apply rotation around center
                x, y = rotate_and_translate(
                    dx, dy, self.center_x, self.center_y, self.yaw
                )
                positions.append((x, y))

        return positions

    def calculate_cluster_4_positions(
        self,
        cluster_count: int,
        cluster_spacing: float,
        cluster_internal_spacing: float,
        wall_length: float,
        wall_thickness: float,
    ) -> list[tuple[float, float]]:
        """Calculate track positions for PPVC_CLUSTER_4 type (2x2 grid per cluster).

        Clusters are placed along wall_width direction.
        Each cluster has 4 rebars in a 2x2 grid pattern.

        Args:
            cluster_count: Number of clusters
            cluster_spacing: Spacing between clusters (meters)
            cluster_internal_spacing: Spacing within cluster (meters)
            wall_length: Length of the wall (meters)
            wall_thickness: Wall thickness for offset (meters)

        Returns:
            List of (x, y) tuples for track positions
        """
        from .structure import Orientation

        positions = []

        # Offset from left edge of wall by wall thickness
        wall_edge_offset = -wall_length / 2 + wall_thickness

        # Clusters are placed along wall_width direction (centered)
        total_span = (cluster_count - 1) * cluster_spacing
        start_offset = -total_span / 2

        for i in range(cluster_count):
            # Cluster center position along wall_width direction
            cluster_center = start_offset + i * cluster_spacing

            # 2x2 grid: 2 rebars in X direction, 2 rebars in Y direction
            for row in range(2):  # Y direction (wall_width)
                for col in range(2):  # X direction (wall_length)
                    # Internal offset within cluster
                    x_offset = (col - 0.5) * cluster_internal_spacing
                    y_offset = (row - 0.5) * cluster_internal_spacing

                    # Position before rotation
                    if self.orientation == Orientation.HORIZONTAL:
                        dx = wall_edge_offset + x_offset
                        dy = cluster_center + y_offset
                    else:
                        dx = cluster_center + y_offset
                        dy = wall_edge_offset + x_offset

                    # Apply rotation around center
                    x, y = rotate_and_translate(
                        dx, dy, self.center_x, self.center_y, self.yaw
                    )
                    positions.append((x, y))

        return positions
