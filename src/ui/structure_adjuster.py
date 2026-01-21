"""Interactive structure position adjuster using matplotlib sliders."""

import math
from dataclasses import dataclass

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np

from ..structure.geometry import rotate_points
from ..config.logging import get_logger
from ..structure.config import StructureConfig

logger = get_logger(__name__)


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
        self._textbox_x: TextBox | None = None
        self._textbox_y: TextBox | None = None
        self._textbox_yaw: TextBox | None = None
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
            logger.warning("Non-interactive backend (%s). Cannot adjust.", backend)
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

        # Create slider axes (narrower to make room for TextBox)
        ax_x = plt.axes([0.15, 0.15, 0.50, 0.03])
        ax_y = plt.axes([0.15, 0.10, 0.50, 0.03])
        ax_yaw = plt.axes([0.15, 0.05, 0.50, 0.03])

        # Create TextBox axes (next to sliders)
        ax_textbox_x = plt.axes([0.66, 0.15, 0.08, 0.03])
        ax_textbox_y = plt.axes([0.66, 0.10, 0.08, 0.03])
        ax_textbox_yaw = plt.axes([0.66, 0.05, 0.08, 0.03])

        # Calculate slider ranges based on data extent
        x_range = self._calculate_range(points[:, 0], structure.center_x)
        y_range = self._calculate_range(points[:, 1], structure.center_y)

        # Create sliders
        self._slider_x = self._create_slider(
            ax_x, "Center X (m)", x_range[0], x_range[1], structure.center_x, 0.001
        )
        self._slider_y = self._create_slider(
            ax_y, "Center Y (m)", y_range[0], y_range[1], structure.center_y, 0.001
        )
        self._slider_yaw = self._create_slider(
            ax_yaw, "Yaw (deg)", -180, 180, math.degrees(structure.yaw), 1
        )

        # Create TextBox widgets
        self._textbox_x = TextBox(
            ax_textbox_x, "", initial=f"{structure.center_x:.3f}"
        )
        self._textbox_y = TextBox(
            ax_textbox_y, "", initial=f"{structure.center_y:.3f}"
        )
        self._textbox_yaw = TextBox(
            ax_textbox_yaw, "", initial=f"{math.degrees(structure.yaw):.1f}"
        )

        # Connect slider callbacks
        self._slider_x.on_changed(self._on_slider_change)
        self._slider_y.on_changed(self._on_slider_change)
        self._slider_yaw.on_changed(self._on_slider_change)

        # Connect TextBox callbacks
        self._textbox_x.on_submit(self._on_textbox_x_submit)
        self._textbox_y.on_submit(self._on_textbox_y_submit)
        self._textbox_yaw.on_submit(self._on_textbox_yaw_submit)

        # Create buttons
        ax_confirm = plt.axes([0.76, 0.10, 0.1, 0.04])
        ax_cancel = plt.axes([0.76, 0.05, 0.1, 0.04])
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
            return (center - 5.0, center + 5.0)
        v_min, v_max = float(np.min(values)), float(np.max(values))
        margin = max(2.0, (v_max - v_min) * 0.5)
        return (min(v_min - margin, center - 5.0), max(v_max + margin, center + 5.0))

    def _create_slider(
        self,
        ax: plt.Axes,
        label: str,
        val_min: float,
        val_max: float,
        valinit: float,
        valstep: float,
    ) -> Slider:
        """Create a slider widget with common settings.

        Args:
            ax: Axes for the slider
            label: Slider label
            val_min: Minimum value
            val_max: Maximum value
            valinit: Initial value
            valstep: Step value

        Returns:
            Configured Slider widget
        """
        return Slider(ax, label, val_min, val_max, valinit=valinit, valstep=valstep)

    def _set_axis_limits(self) -> None:
        """Set axis limits based on points and structure."""
        if self._points is None or len(self._points) == 0:
            return

        cfg = self._structure_config
        if cfg is None:
            return

        display_width, display_height = cfg.get_display_dimensions()
        half_w, half_h = display_width / 2, display_height / 2

        x_min = min(np.min(self._points[:, 0]), cfg.center_x - half_w)
        x_max = max(np.max(self._points[:, 0]), cfg.center_x + half_w)
        y_min = min(np.min(self._points[:, 1]), cfg.center_y - half_h)
        y_max = max(np.max(self._points[:, 1]), cfg.center_y + half_h)

        margin = 0.1
        self._ax.set_xlim(x_min - margin, x_max + margin)
        self._ax.set_ylim(y_min - margin, y_max + margin)

    def _on_slider_change(self, val) -> None:
        """Handle slider value change - update structure drawing and TextBoxes."""
        self._position = StructurePosition(
            center_x=self._slider_x.val,
            center_y=self._slider_y.val,
            yaw=math.radians(self._slider_yaw.val),
        )
        # Update TextBox values to match sliders
        if self._textbox_x is not None:
            self._textbox_x.set_val(f"{self._slider_x.val:.3f}")
        if self._textbox_y is not None:
            self._textbox_y.set_val(f"{self._slider_y.val:.3f}")
        if self._textbox_yaw is not None:
            self._textbox_yaw.set_val(f"{self._slider_yaw.val:.1f}")
        self._draw_structure()
        self._fig.canvas.draw_idle()

    def _on_textbox_x_submit(self, text: str) -> None:
        """Handle X TextBox value submission."""
        try:
            value = float(text)
            self._slider_x.set_val(value)
        except ValueError:
            # Invalid input - reset to current slider value
            if self._textbox_x is not None:
                self._textbox_x.set_val(f"{self._slider_x.val:.3f}")

    def _on_textbox_y_submit(self, text: str) -> None:
        """Handle Y TextBox value submission."""
        try:
            value = float(text)
            self._slider_y.set_val(value)
        except ValueError:
            # Invalid input - reset to current slider value
            if self._textbox_y is not None:
                self._textbox_y.set_val(f"{self._slider_y.val:.3f}")

    def _on_textbox_yaw_submit(self, text: str) -> None:
        """Handle Yaw TextBox value submission."""
        try:
            value = float(text)
            # Clamp to slider range
            value = max(-180, min(180, value))
            self._slider_yaw.set_val(value)
        except ValueError:
            # Invalid input - reset to current slider value
            if self._textbox_yaw is not None:
                self._textbox_yaw.set_val(f"{self._slider_yaw.val:.1f}")

    def _draw_structure(self) -> None:
        """Draw/redraw structure overlay with current position."""
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

        # Temporarily update config position for get_track_positions()
        original_x, original_y, original_yaw = cfg.center_x, cfg.center_y, cfg.yaw
        cfg.center_x = pos.center_x
        cfg.center_y = pos.center_y
        cfg.yaw = pos.yaw

        display_width, display_height = cfg.get_display_dimensions()
        half_w, half_h = display_width / 2, display_height / 2

        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

        rotated = rotate_points(corners, pos.center_x, pos.center_y, pos.yaw)

        poly = mpatches.Polygon(
            rotated,
            fill=False,
            edgecolor="gray",
            linewidth=2,
            linestyle="-",
        )
        self._ax.add_patch(poly)
        self._structure_patches.append(poly)

        track_positions = cfg.get_track_positions()
        track_radius = cfg.track_diameter / 2

        for x, y in track_positions:
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

        cfg.center_x = original_x
        cfg.center_y = original_y
        cfg.yaw = original_yaw

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
