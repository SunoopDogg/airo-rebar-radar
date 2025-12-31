"""Interactive structure position adjuster using matplotlib sliders."""

import math
from dataclasses import dataclass

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from .config import StructureConfig


@dataclass
class StructurePosition:
    """Structure position values."""

    center_x: float
    center_y: float
    yaw: float  # radians

    def yaw_degrees(self) -> float:
        """Return yaw in degrees."""
        return math.degrees(self.yaw)


class StructureAdjuster:
    """Interactive structure position adjuster using matplotlib sliders."""

    def __init__(self):
        """Initialize the structure adjuster."""
        self._position: StructurePosition | None = None
        self._confirmed: bool = False
        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None

        # Widget references
        self._slider_x: Slider | None = None
        self._slider_y: Slider | None = None
        self._slider_yaw: Slider | None = None
        self._btn_confirm: Button | None = None
        self._btn_cancel: Button | None = None

        # Drawing elements (for live update)
        self._structure_patches: list = []
        self._structure_config: StructureConfig | None = None
        self._points: np.ndarray | None = None

    def adjust_structure(
        self,
        points: np.ndarray,
        structure: StructureConfig,
        title: str = "Structure Position Adjustment",
    ) -> StructurePosition | None:
        """Show interactive UI for structure position adjustment.

        Args:
            points: Array of shape (N, 2) with x, y coordinates for context
            structure: Initial structure configuration
            title: Window title

        Returns:
            StructurePosition if confirmed, None if cancelled
        """
        # Check for non-interactive backend
        backend = matplotlib.get_backend()
        if backend.lower() in ("agg", "pdf", "svg", "ps"):
            print(f"Warning: Non-interactive backend ({backend}). Cannot adjust.")
            return None

        self._structure_config = structure
        self._points = points
        self._confirmed = False
        self._position = StructurePosition(
            center_x=structure.center_x,
            center_y=structure.center_y,
            yaw=structure.yaw,
        )

        # Create figure with space for sliders
        self._fig, self._ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.25)  # Make room for sliders

        # Plot background points
        if len(points) > 0:
            self._ax.scatter(
                points[:, 0],
                points[:, 1],
                s=3,
                alpha=0.5,
                c="lightgray",
                label="LIDAR Points",
            )

        # Initial structure drawing
        self._draw_structure()

        # Setup axes
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_title(title)
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.3)

        # Set axis limits based on points and structure
        self._set_axis_limits()

        # Create slider axes
        ax_x = plt.axes([0.15, 0.15, 0.55, 0.03])
        ax_y = plt.axes([0.15, 0.10, 0.55, 0.03])
        ax_yaw = plt.axes([0.15, 0.05, 0.55, 0.03])

        # Calculate slider ranges based on data extent
        x_range = self._calculate_range(points[:, 0], structure.center_x)
        y_range = self._calculate_range(points[:, 1], structure.center_y)

        # Create sliders
        self._slider_x = Slider(
            ax_x,
            "Center X (m)",
            x_range[0],
            x_range[1],
            valinit=structure.center_x,
            valstep=0.01,
        )
        self._slider_y = Slider(
            ax_y,
            "Center Y (m)",
            y_range[0],
            y_range[1],
            valinit=structure.center_y,
            valstep=0.01,
        )
        self._slider_yaw = Slider(
            ax_yaw,
            "Yaw (deg)",
            -180,
            180,
            valinit=math.degrees(structure.yaw),
            valstep=1,
        )

        # Connect slider callbacks
        self._slider_x.on_changed(self._on_slider_change)
        self._slider_y.on_changed(self._on_slider_change)
        self._slider_yaw.on_changed(self._on_slider_change)

        # Create buttons
        ax_confirm = plt.axes([0.75, 0.10, 0.1, 0.04])
        ax_cancel = plt.axes([0.75, 0.05, 0.1, 0.04])
        self._btn_confirm = Button(ax_confirm, "Confirm")
        self._btn_cancel = Button(ax_cancel, "Cancel")

        self._btn_confirm.on_clicked(self._on_confirm)
        self._btn_cancel.on_clicked(self._on_cancel)

        # Keyboard shortcuts
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Add instruction text
        self._ax.text(
            0.5,
            0.02,
            "Use sliders to adjust | ENTER: confirm | ESC: cancel",
            transform=self._ax.transAxes,
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            fontsize=10,
        )

        # Show (blocking)
        plt.show()

        if self._confirmed:
            return self._position
        return None

    def _calculate_range(
        self, values: np.ndarray, center: float
    ) -> tuple[float, float]:
        """Calculate appropriate slider range."""
        if len(values) == 0:
            return (center - 2.0, center + 2.0)
        v_min, v_max = float(np.min(values)), float(np.max(values))
        margin = max(0.5, (v_max - v_min) * 0.2)
        return (min(v_min - margin, center - 1.0), max(v_max + margin, center + 1.0))

    def _set_axis_limits(self) -> None:
        """Set axis limits based on points and structure."""
        if self._points is None or len(self._points) == 0:
            return

        cfg = self._structure_config
        if cfg is None:
            return

        # Get structure extent
        display_width, display_height = cfg.get_display_dimensions()
        half_w, half_h = display_width / 2, display_height / 2

        # Calculate bounds including structure and points
        x_min = min(np.min(self._points[:, 0]), cfg.center_x - half_w)
        x_max = max(np.max(self._points[:, 0]), cfg.center_x + half_w)
        y_min = min(np.min(self._points[:, 1]), cfg.center_y - half_h)
        y_max = max(np.max(self._points[:, 1]), cfg.center_y + half_h)

        margin = 0.1
        self._ax.set_xlim(x_min - margin, x_max + margin)
        self._ax.set_ylim(y_min - margin, y_max + margin)

    def _on_slider_change(self, val) -> None:
        """Handle slider value change - update structure drawing."""
        self._position = StructurePosition(
            center_x=self._slider_x.val,
            center_y=self._slider_y.val,
            yaw=math.radians(self._slider_yaw.val),
        )
        self._draw_structure()
        self._fig.canvas.draw_idle()

    def _draw_structure(self) -> None:
        """Draw/redraw structure overlay with current position."""
        # Remove previous patches
        for patch in self._structure_patches:
            try:
                patch.remove()
            except ValueError:
                pass  # Already removed
        self._structure_patches.clear()

        if self._structure_config is None or self._position is None:
            return

        cfg = self._structure_config
        pos = self._position

        # Get dimensions
        display_width, display_height = cfg.get_display_dimensions()
        half_w, half_h = display_width / 2, display_height / 2

        # Compute rotated rectangle corners
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

        cos_y, sin_y = math.cos(pos.yaw), math.sin(pos.yaw)
        rotated = []
        for dx, dy in corners:
            rx = dx * cos_y - dy * sin_y + pos.center_x
            ry = dx * sin_y + dy * cos_y + pos.center_y
            rotated.append((rx, ry))

        # Draw concrete outline
        poly = mpatches.Polygon(
            rotated,
            fill=False,
            edgecolor="gray",
            linewidth=2,
            linestyle="-",
        )
        self._ax.add_patch(poly)
        self._structure_patches.append(poly)

        # Draw expected rebar positions (apply rotation to track positions)
        spacing_x, spacing_y = cfg.get_display_track_spacing()
        track_radius = cfg.track_diameter / 2

        for i in range(cfg.track_count_x):
            for j in range(cfg.track_count_y):
                dx = (i - 0.5) * spacing_x if cfg.track_count_x == 2 else 0.0
                dy = (j - 0.5) * spacing_y if cfg.track_count_y == 2 else 0.0

                x = dx * cos_y - dy * sin_y + pos.center_x
                y = dx * sin_y + dy * cos_y + pos.center_y

                circle = plt.Circle(
                    (x, y),
                    track_radius,
                    fill=False,
                    edgecolor="blue",
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.7,
                )
                self._ax.add_patch(circle)
                self._structure_patches.append(circle)

                (marker,) = self._ax.plot(
                    x,
                    y,
                    "+",
                    color="blue",
                    markersize=8,
                    markeredgewidth=1.5,
                    alpha=0.7,
                )
                self._structure_patches.append(marker)

    def _on_confirm(self, event) -> None:
        """Handle confirm button click."""
        self._confirmed = True
        plt.close(self._fig)

    def _on_cancel(self, event) -> None:
        """Handle cancel button click."""
        self._confirmed = False
        plt.close(self._fig)

    def _on_key(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key == "enter":
            self._on_confirm(event)
        elif event.key == "escape":
            self._on_cancel(event)
