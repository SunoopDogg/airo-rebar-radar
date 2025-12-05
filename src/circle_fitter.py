"""Circle fitting module using Least Squares method."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .utils.config import CircleFittingConfig


@dataclass
class CircleFitResult:
    """Result of circle fitting."""
    center_x: float
    center_y: float
    radius: float
    residual: float
    num_points: int
    success: bool


class CircleFitter:
    """Fit circles to point clusters using Least Squares."""

    def __init__(self, config: CircleFittingConfig | None = None):
        """Initialize circle fitter.

        Args:
            config: Circle fitting configuration
        """
        self.config = config or CircleFittingConfig()

    def _algebraic_circle_fit(self, points: np.ndarray) -> tuple[float, float, float]:
        """Algebraic circle fit using Kasa method.

        This provides a fast initial estimate for the iterative method.

        Args:
            points: Array of shape (N, 2) with x, y coordinates

        Returns:
            Tuple of (center_x, center_y, radius)
        """
        x = points[:, 0]
        y = points[:, 1]
        n = len(points)

        # Build the system matrix A and vector b
        # Circle equation: (x - cx)^2 + (y - cy)^2 = r^2
        # Expanded: x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = 0
        # Let: A*cx + B*cy + C = -(x^2 + y^2) where A=2x, B=2y, C=1

        A = np.column_stack([2 * x, 2 * y, np.ones(n)])
        b = x**2 + y**2

        # Solve using least squares
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        cx, cy, c = result
        radius = np.sqrt(c + cx**2 + cy**2)

        return cx, cy, radius

    def _geometric_residuals(
        self,
        params: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """Calculate geometric residuals for circle fit.

        Args:
            params: Array [cx, cy, r]
            points: Array of shape (N, 2)

        Returns:
            Array of residuals (distance to circle)
        """
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        return distances - r

    def fit_circle(self, points: np.ndarray) -> CircleFitResult:
        """Fit a circle to a set of points.

        Uses algebraic fit for initial estimate, then refines with
        geometric least squares optimization.

        Args:
            points: Array of shape (N, 2) with x, y coordinates

        Returns:
            CircleFitResult with fitted parameters
        """
        n_points = len(points)

        # Check minimum points requirement
        if n_points < self.config.min_points:
            return CircleFitResult(
                center_x=0, center_y=0, radius=0,
                residual=float("inf"), num_points=n_points, success=False
            )

        # Get initial estimate using algebraic method
        try:
            cx0, cy0, r0 = self._algebraic_circle_fit(points)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to centroid if algebraic fit fails
            cx0 = np.mean(points[:, 0])
            cy0 = np.mean(points[:, 1])
            r0 = np.mean(np.sqrt(
                (points[:, 0] - cx0)**2 + (points[:, 1] - cy0)**2
            ))

        # Ensure positive radius
        r0 = max(abs(r0), 0.001)

        # Refine with geometric least squares
        try:
            result = least_squares(
                self._geometric_residuals,
                x0=[cx0, cy0, r0],
                args=(points,),
                bounds=(
                    [-np.inf, -np.inf, self.config.min_radius],
                    [np.inf, np.inf, self.config.max_radius]
                ),
                method="trf"
            )

            cx, cy, r = result.x
            residual = np.mean(result.fun**2)  # MSE
            success = result.success

        except Exception:
            # Fall back to algebraic solution
            cx, cy, r = cx0, cy0, r0
            residuals = self._geometric_residuals([cx, cy, r], points)
            residual = np.mean(residuals**2)
            success = False

        # Validate radius bounds
        if not (self.config.min_radius <= r <= self.config.max_radius):
            success = False

        return CircleFitResult(
            center_x=float(cx),
            center_y=float(cy),
            radius=float(r),
            residual=float(residual),
            num_points=n_points,
            success=success
        )

    def fit_clusters(
        self,
        clusters: list[np.ndarray]
    ) -> list[CircleFitResult]:
        """Fit circles to multiple clusters.

        Args:
            clusters: List of point arrays

        Returns:
            List of CircleFitResult for each cluster
        """
        results = []
        for cluster in clusters:
            result = self.fit_circle(cluster)
            results.append(result)
        return results
