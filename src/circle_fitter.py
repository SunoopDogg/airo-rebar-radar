"""Circle fitting module using 3-point combinatorial method."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .utils.config import CircleFittingConfig
from .utils.geometry import calculate_distance_np
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CircleFitResult:
    """Result of circle fitting."""
    center_x: float
    center_y: float
    radius: float
    residual: float
    num_points: int


class CircleFitter:
    """Fit circles to point clusters using 3-point combinatorial method."""

    def __init__(self, config: CircleFittingConfig | None = None):
        """Initialize circle fitter.

        Args:
            config: Circle fitting configuration
        """
        self.config = config or CircleFittingConfig()

    def _compute_circumcircle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Calculate circumscribed circle from 3 points.

        Uses the circumcenter formula to find the unique circle
        passing through all three points.

        Args:
            p1, p2, p3: Points as numpy arrays [x, y]

        Returns:
            Tuple of (center_x, center_y, radius) or None if collinear
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # D = 2 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
        denominator = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        if abs(denominator) < self.config.collinearity_tolerance:
            return None

        # ux = ((x1²+y1²)(y2-y3) + (x2²+y2²)(y3-y1) + (x3²+y3²)(y1-y2)) / D
        # uy = ((x1²+y1²)(x3-x2) + (x2²+y2²)(x1-x3) + (x3²+y3²)(x2-x1)) / D
        mag_sq_1 = x1**2 + y1**2
        mag_sq_2 = x2**2 + y2**2
        mag_sq_3 = x3**2 + y3**2

        cx = (mag_sq_1 * (y2 - y3) + mag_sq_2 * (y3 - y1) + mag_sq_3 * (y1 - y2)) / denominator
        cy = (mag_sq_1 * (x3 - x2) + mag_sq_2 * (x1 - x3) + mag_sq_3 * (x2 - x1)) / denominator

        radius = calculate_distance_np(x1, y1, cx, cy)

        return cx, cy, radius

    def fit_circle(self, points: np.ndarray) -> CircleFitResult:
        """Fit a circle using 3-point combinatorial method with averaging.

        Exhaustively searches all 3-point combinations, collects all circles
        within the radius range, and returns the average circle.

        Args:
            points: Array of shape (N, 2) with x, y coordinates

        Returns:
            CircleFitResult with averaged parameters
        """
        n_points = len(points)

        # Check minimum points requirement
        if n_points < self.config.min_points:
            logger.debug(
                "Insufficient points for circle fitting: %d < %d",
                n_points, self.config.min_points
            )
            return CircleFitResult(
                center_x=0, center_y=0, radius=0,
                residual=float("inf"), num_points=n_points
            )

        valid_circles = []

        for i, j, k in combinations(range(n_points), 3):
            p1, p2, p3 = points[i], points[j], points[k]

            result = self._compute_circumcircle(p1, p2, p3)
            if result is None:
                continue

            cx, cy, radius = result

            if self.config.min_radius <= radius <= self.config.max_radius:
                valid_circles.append((cx, cy, radius))

        if not valid_circles:
            logger.debug(
                "No valid circles found within radius range [%.4f, %.4f]m "
                "from %d points",
                self.config.min_radius, self.config.max_radius, n_points
            )
            return CircleFitResult(
                center_x=0, center_y=0, radius=0,
                residual=float("inf"), num_points=n_points
            )

        # Calculate average of all valid circles (vectorized)
        circles_array = np.array(valid_circles)
        means = np.mean(circles_array, axis=0)
        stds = np.std(circles_array, axis=0)

        avg_cx, avg_cy, avg_radius = float(means[0]), float(means[1]), float(means[2])
        residual = float(np.sum(stds))

        return CircleFitResult(
            center_x=avg_cx,
            center_y=avg_cy,
            radius=avg_radius,
            residual=residual,
            num_points=n_points,
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

            if result.radius == 0:
                continue

            results.append(result)
        return results
