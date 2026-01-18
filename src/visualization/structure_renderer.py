"""Structure rendering for visualization."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ..structure.geometry import rotate_points
from ..structure.config import StructureConfig


class StructureRenderer:
    """Render structure overlays on matplotlib axes."""

    def __init__(self, structure: StructureConfig):
        """Initialize the renderer.

        Args:
            structure: Structure configuration to render
        """
        self.structure = structure

    def draw_boundary(self, ax: plt.Axes) -> mpatches.Polygon:
        """Draw the structure boundary (concrete outline).

        Args:
            ax: Matplotlib axes to draw on

        Returns:
            The polygon patch that was added
        """
        display_width, display_height = self.structure.get_display_dimensions()

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
        rotated_corners = rotate_points(
            corners,
            self.structure.center_x,
            self.structure.center_y,
            self.structure.yaw,
        )

        poly = mpatches.Polygon(
            rotated_corners,
            fill=False,
            edgecolor="gray",
            linewidth=2,
            linestyle="-",
            label="Concrete outline",
        )
        ax.add_patch(poly)
        return poly

    def draw_expected_rebars(
        self,
        ax: plt.Axes,
        color: str = "blue",
        linestyle: str = ":",
        alpha: float = 0.7,
    ) -> list[plt.Circle]:
        """Draw expected rebar positions.

        Args:
            ax: Matplotlib axes to draw on
            color: Color for rebar markers
            linestyle: Line style for rebar circles
            alpha: Transparency value

        Returns:
            List of circle patches that were added
        """
        track_positions = self.structure.get_track_positions()
        track_radius = self.structure.track_diameter / 2

        circles = []
        for x, y in track_positions:
            circle = plt.Circle(
                (x, y),
                track_radius,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                linestyle=linestyle,
                alpha=alpha,
            )
            ax.add_patch(circle)
            circles.append(circle)

            ax.plot(
                x, y,
                "+",
                color=color,
                markersize=8,
                markeredgewidth=1.5,
                alpha=alpha,
            )

        return circles

    def draw_full_overlay(self, ax: plt.Axes) -> None:
        """Draw complete structure overlay (boundary + expected rebars).

        Args:
            ax: Matplotlib axes to draw on
        """
        self.draw_boundary(ax)
        self.draw_expected_rebars(ax)

    def get_axis_limits(
        self,
        margin: float | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calculate axis limits based on structure dimensions.

        Args:
            margin: Optional margin to add around structure.
                    If None, uses 10% of the larger dimension or 0.1m minimum.

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max))
        """
        display_width, display_height = self.structure.get_display_dimensions()

        if margin is None:
            margin = max(0.1, max(display_width, display_height) * 0.1)

        x_limits = (
            self.structure.center_x - display_width / 2 - margin,
            self.structure.center_x + display_width / 2 + margin,
        )
        y_limits = (
            self.structure.center_y - display_height / 2 - margin,
            self.structure.center_y + display_height / 2 + margin,
        )

        return x_limits, y_limits

    def set_axis_limits(
        self,
        ax: plt.Axes,
        margin: float | None = None,
    ) -> None:
        """Set axis limits on the given axes.

        Args:
            ax: Matplotlib axes to set limits on
            margin: Optional margin to add around structure
        """
        x_limits, y_limits = self.get_axis_limits(margin)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)


def create_structure_legend_elements() -> list:
    """Create legend elements for structure overlay.

    Returns:
        List of legend elements for matplotlib
    """
    return [
        plt.Line2D(
            [0], [0],
            color="gray",
            linestyle="-",
            linewidth=2,
            label="Concrete outline",
        ),
        plt.Line2D(
            [0], [0],
            color="blue",
            linestyle=":",
            linewidth=1.5,
            label="Expected rebar",
        ),
        plt.Line2D(
            [0], [0],
            color="red",
            linestyle="-",
            linewidth=2,
            label="Detected rebar",
        ),
    ]
