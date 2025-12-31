"""Interactive ROI selector using matplotlib RectangleSelector."""

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np


@dataclass
class ROIBounds:
    """Selected ROI boundary values."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def is_valid(self, min_dimension: float = 0.01) -> bool:
        """Check if ROI bounds are valid (positive width and height).

        Args:
            min_dimension: Minimum dimension in meters (default 1cm)

        Returns:
            True if ROI has positive area above minimum threshold
        """
        return (
            (self.x_max - self.x_min) >= min_dimension
            and (self.y_max - self.y_min) >= min_dimension
        )


class ROISelector:
    """Interactive ROI selector using matplotlib RectangleSelector."""

    def __init__(self):
        """Initialize the ROI selector."""
        self._selected_bounds: ROIBounds | None = None
        self._confirmed: bool = False
        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None
        self._rect_selector: RectangleSelector | None = None
        self._status_text = None
        self._points: np.ndarray | None = None

    def select_roi(
        self,
        points: np.ndarray,
        title: str = "ROI Selection",
        default_bounds: ROIBounds | None = None,
    ) -> ROIBounds | None:
        """Show point cloud and allow user to select ROI.

        Args:
            points: Array of shape (N, 2) with x, y coordinates
            title: Window title
            default_bounds: Optional default ROI to show initially

        Returns:
            ROIBounds if confirmed, None if cancelled
        """
        # Check for non-interactive backend
        backend = matplotlib.get_backend()
        if backend.lower() in ("agg", "pdf", "svg", "ps"):
            print(f"Warning: Non-interactive backend ({backend}). Using default ROI.")
            return None

        if len(points) == 0:
            print("Warning: No points available for ROI preview. Using default ROI.")
            return None

        self._points = points
        self._confirmed = False
        self._selected_bounds = default_bounds

        # Create figure
        self._fig, self._ax = plt.subplots(figsize=(12, 10))

        # Plot points
        self._ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.6, c="blue")
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_title(title)
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.3)

        # Add instruction text at bottom
        instruction = "Drag to select ROI | ENTER: confirm | ESC: cancel | R: reset"
        self._ax.text(
            0.5,
            0.02,
            instruction,
            transform=self._ax.transAxes,
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            fontsize=10,
        )

        # Initialize RectangleSelector
        self._rect_selector = RectangleSelector(
            self._ax,
            self._on_select,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=0.01,
            minspany=0.01,
            spancoords="data",
            interactive=True,
            props={"facecolor": "yellow", "alpha": 0.3, "edgecolor": "red", "linewidth": 2},
        )

        # Draw default bounds if provided
        if default_bounds and default_bounds.is_valid():
            self._rect_selector.extents = (
                default_bounds.x_min,
                default_bounds.x_max,
                default_bounds.y_min,
                default_bounds.y_max,
            )
            self._update_status()

        # Connect keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Show (blocking)
        plt.show()

        # Return result
        if self._confirmed and self._selected_bounds and self._selected_bounds.is_valid():
            return self._selected_bounds
        return None

    def _on_select(self, eclick, erelease) -> None:
        """Callback when rectangle selection is made."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Ensure min/max ordering
        self._selected_bounds = ROIBounds(
            x_min=min(x1, x2),
            x_max=max(x1, x2),
            y_min=min(y1, y2),
            y_max=max(y1, y2),
        )

        # Update status display
        self._update_status()

    def _on_key(self, event) -> None:
        """Handle keyboard events for confirm/cancel."""
        if event.key == "enter":
            if self._selected_bounds and self._selected_bounds.is_valid():
                self._confirmed = True
                plt.close(self._fig)
            else:
                print("Please select a valid ROI before confirming.")

        elif event.key == "escape":
            self._confirmed = False
            plt.close(self._fig)

        elif event.key == "r":
            # Reset selection
            self._selected_bounds = None
            if self._rect_selector:
                self._rect_selector.set_active(True)
            self._update_status()

    def _update_status(self) -> None:
        """Update the status text showing current ROI bounds and point count."""
        # Remove existing status text
        if self._status_text:
            self._status_text.remove()
            self._status_text = None

        if self._selected_bounds:
            b = self._selected_bounds
            # Count points inside selected area
            if self._points is not None:
                mask = (
                    (self._points[:, 0] >= b.x_min)
                    & (self._points[:, 0] <= b.x_max)
                    & (self._points[:, 1] >= b.y_min)
                    & (self._points[:, 1] <= b.y_max)
                )
                n_inside = int(np.sum(mask))
                text = (
                    f"ROI: X=[{b.x_min:.3f}, {b.x_max:.3f}]m, "
                    f"Y=[{b.y_min:.3f}, {b.y_max:.3f}]m | "
                    f"Points: {n_inside}/{len(self._points)}"
                )
            else:
                text = (
                    f"ROI: X=[{b.x_min:.3f}, {b.x_max:.3f}]m, "
                    f"Y=[{b.y_min:.3f}, {b.y_max:.3f}]m"
                )
        else:
            text = "No ROI selected"

        self._status_text = self._ax.text(
            0.5,
            0.98,
            text,
            transform=self._ax.transAxes,
            ha="center",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.8},
            fontsize=10,
        )
        self._fig.canvas.draw_idle()
