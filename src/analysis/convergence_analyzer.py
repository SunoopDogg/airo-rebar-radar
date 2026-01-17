"""Convergence analysis module for interval_averages based track stabilization."""

from dataclasses import dataclass, field
from math import sqrt

from ..core.temporal_filter import Track


@dataclass
class ConvergenceThresholds:
    """Thresholds for determining convergence."""
    position_threshold_mm: float = 0.1
    radius_threshold_mm: float = 0.01
    consecutive_intervals: int = 3


@dataclass
class ConvergencePoint:
    """Convergence point information for a single track."""
    track_id: int
    position_converged: bool
    position_convergence_interval: int | None
    radius_converged: bool
    radius_convergence_interval: int | None
    final_position_change_mm: float | None
    final_radius_change_mm: float | None


@dataclass
class ConvergenceMetrics:
    """Convergence metrics for a single track."""
    track_id: int
    intervals: list[int] = field(default_factory=list)
    center_x_values: list[float] = field(default_factory=list)
    center_y_values: list[float] = field(default_factory=list)
    radius_values: list[float] = field(default_factory=list)
    position_convergence: list[float] = field(default_factory=list)
    radius_convergence: list[float] = field(default_factory=list)


class ConvergenceAnalyzer:
    """Analyzer for interval_averages based convergence metrics."""

    def analyze_track(self, track: Track) -> ConvergenceMetrics:
        """Analyze convergence of a single track.

        Args:
            track: Track object with interval_averages data

        Returns:
            ConvergenceMetrics with calculated convergence rates
        """
        interval_averages = track.interval_averages
        if not interval_averages:
            return ConvergenceMetrics(track_id=track.track_id)

        intervals = sorted(interval_averages.keys())

        # Extract values
        center_x_values = [interval_averages[i]["center_x"] for i in intervals]
        center_y_values = [interval_averages[i]["center_y"] for i in intervals]
        radius_values = [interval_averages[i]["radius"] for i in intervals]

        # Calculate convergence rates
        position_convergence: list[float] = []
        radius_convergence: list[float] = []

        for i in range(1, len(intervals)):
            # Position change (Euclidean distance in mm)
            dx = center_x_values[i] - center_x_values[i - 1]
            dy = center_y_values[i] - center_y_values[i - 1]
            pos_change_mm = sqrt(dx ** 2 + dy ** 2) * 1000  # Convert m to mm
            position_convergence.append(pos_change_mm)

            # Radius change (mm)
            dr = abs(radius_values[i] - radius_values[i - 1])
            rad_change_mm = dr * 1000  # Convert m to mm
            radius_convergence.append(rad_change_mm)

        return ConvergenceMetrics(
            track_id=track.track_id,
            intervals=intervals,
            center_x_values=center_x_values,
            center_y_values=center_y_values,
            radius_values=radius_values,
            position_convergence=position_convergence,
            radius_convergence=radius_convergence,
        )

    def analyze_tracks(self, tracks: list[Track]) -> list[ConvergenceMetrics]:
        """Analyze convergence of multiple tracks.

        Args:
            tracks: List of Track objects

        Returns:
            List of ConvergenceMetrics for each track
        """
        return [self.analyze_track(track) for track in tracks]

    def find_convergence_point(
        self,
        metrics: ConvergenceMetrics,
        thresholds: ConvergenceThresholds,
    ) -> ConvergencePoint:
        """Find the convergence point for a single track.

        Convergence is determined when N consecutive intervals have changes
        below the threshold.

        Args:
            metrics: ConvergenceMetrics for a track
            thresholds: ConvergenceThresholds configuration

        Returns:
            ConvergencePoint with convergence information
        """
        n_consecutive = thresholds.consecutive_intervals
        pos_threshold = thresholds.position_threshold_mm
        rad_threshold = thresholds.radius_threshold_mm

        position_convergence_interval: int | None = None
        radius_convergence_interval: int | None = None

        # Find position convergence point
        pos_changes = metrics.position_convergence
        if len(pos_changes) >= n_consecutive:
            for i in range(len(pos_changes) - n_consecutive + 1):
                window = pos_changes[i:i + n_consecutive]
                if all(change < pos_threshold for change in window):
                    # Convergence starts at the interval after the first change in the window
                    position_convergence_interval = metrics.intervals[i + 1]
                    break

        # Find radius convergence point
        rad_changes = metrics.radius_convergence
        if len(rad_changes) >= n_consecutive:
            for i in range(len(rad_changes) - n_consecutive + 1):
                window = rad_changes[i:i + n_consecutive]
                if all(change < rad_threshold for change in window):
                    radius_convergence_interval = metrics.intervals[i + 1]
                    break

        # Get final change values
        final_position_change = pos_changes[-1] if pos_changes else None
        final_radius_change = rad_changes[-1] if rad_changes else None

        return ConvergencePoint(
            track_id=metrics.track_id,
            position_converged=position_convergence_interval is not None,
            position_convergence_interval=position_convergence_interval,
            radius_converged=radius_convergence_interval is not None,
            radius_convergence_interval=radius_convergence_interval,
            final_position_change_mm=final_position_change,
            final_radius_change_mm=final_radius_change,
        )

    def find_all_convergence_points(
        self,
        metrics_list: list[ConvergenceMetrics],
        thresholds: ConvergenceThresholds,
    ) -> list[ConvergencePoint]:
        """Find convergence points for all tracks.

        Args:
            metrics_list: List of ConvergenceMetrics
            thresholds: ConvergenceThresholds configuration

        Returns:
            List of ConvergencePoint for each track
        """
        return [
            self.find_convergence_point(metrics, thresholds)
            for metrics in metrics_list
        ]

    def find_optimal_thresholds(
        self,
        metrics_list: list[ConvergenceMetrics],
        consecutive_intervals: int = 3,
    ) -> ConvergenceThresholds:
        """Find optimal thresholds where all tracks converge.

        For each track, finds the minimum "max change in window" across all
        consecutive windows. The optimal threshold is the maximum of these
        values across all tracks, ensuring all tracks can converge.

        Args:
            metrics_list: List of ConvergenceMetrics
            consecutive_intervals: Number of consecutive intervals for convergence

        Returns:
            ConvergenceThresholds with optimal values for all tracks to converge
        """
        position_min_maxes: list[float] = []
        radius_min_maxes: list[float] = []

        for metrics in metrics_list:
            pos_changes = metrics.position_convergence
            rad_changes = metrics.radius_convergence

            # Find minimum of maximum values in consecutive windows for position
            if len(pos_changes) >= consecutive_intervals:
                min_window_max = float('inf')
                for i in range(len(pos_changes) - consecutive_intervals + 1):
                    window = pos_changes[i:i + consecutive_intervals]
                    window_max = max(window)
                    min_window_max = min(min_window_max, window_max)
                position_min_maxes.append(min_window_max)

            # Find minimum of maximum values in consecutive windows for radius
            if len(rad_changes) >= consecutive_intervals:
                min_window_max = float('inf')
                for i in range(len(rad_changes) - consecutive_intervals + 1):
                    window = rad_changes[i:i + consecutive_intervals]
                    window_max = max(window)
                    min_window_max = min(min_window_max, window_max)
                radius_min_maxes.append(min_window_max)

        # Optimal threshold is the max of all min-maxes (to ensure ALL tracks converge)
        # Add small epsilon to ensure strict inequality works
        epsilon = 1e-9
        optimal_position = (
            max(position_min_maxes) + epsilon
            if position_min_maxes
            else 0.1  # default
        )
        optimal_radius = (
            max(radius_min_maxes) + epsilon
            if radius_min_maxes
            else 0.01  # default
        )

        return ConvergenceThresholds(
            position_threshold_mm=optimal_position,
            radius_threshold_mm=optimal_radius,
            consecutive_intervals=consecutive_intervals,
        )

    def get_summary(
        self,
        metrics: list[ConvergenceMetrics],
        thresholds: ConvergenceThresholds | None = None,
    ) -> dict:
        """Generate summary statistics for convergence metrics.

        Args:
            metrics: List of ConvergenceMetrics
            thresholds: Optional thresholds for convergence point detection

        Returns:
            Dictionary with summary statistics including convergence info
        """
        if thresholds is None:
            thresholds = ConvergenceThresholds()

        convergence_points = self.find_all_convergence_points(metrics, thresholds)

        # Calculate convergence statistics
        position_converged = [cp for cp in convergence_points if cp.position_converged]
        radius_converged = [cp for cp in convergence_points if cp.radius_converged]
        fully_converged = [
            cp for cp in convergence_points
            if cp.position_converged and cp.radius_converged
        ]

        # Calculate average convergence intervals
        pos_intervals = [
            cp.position_convergence_interval
            for cp in position_converged
            if cp.position_convergence_interval is not None
        ]
        rad_intervals = [
            cp.radius_convergence_interval
            for cp in radius_converged
            if cp.radius_convergence_interval is not None
        ]

        avg_pos_interval = sum(pos_intervals) / len(pos_intervals) if pos_intervals else None
        avg_rad_interval = sum(rad_intervals) / len(rad_intervals) if rad_intervals else None

        # Calculate optimal thresholds
        optimal = self.find_optimal_thresholds(metrics, thresholds.consecutive_intervals)

        return {
            "total_count": len(metrics),
            "thresholds": {
                "position_threshold_mm": thresholds.position_threshold_mm,
                "radius_threshold_mm": thresholds.radius_threshold_mm,
                "consecutive_intervals": thresholds.consecutive_intervals,
            },
            "summary": {
                "position_converged_count": len(position_converged),
                "radius_converged_count": len(radius_converged),
                "fully_converged_count": len(fully_converged),
                "avg_position_convergence_interval": avg_pos_interval,
                "avg_radius_convergence_interval": avg_rad_interval,
            },
            "optimal_thresholds": {
                "position_threshold_mm": optimal.position_threshold_mm,
                "radius_threshold_mm": optimal.radius_threshold_mm,
                "consecutive_intervals": optimal.consecutive_intervals,
            },
            "tracks": [
                {
                    "track_id": cp.track_id,
                    "position_converged": cp.position_converged,
                    "position_convergence_interval": cp.position_convergence_interval,
                    "radius_converged": cp.radius_converged,
                    "radius_convergence_interval": cp.radius_convergence_interval,
                    "final_position_change_mm": cp.final_position_change_mm,
                    "final_radius_change_mm": cp.final_radius_change_mm,
                }
                for cp in convergence_points
            ],
        }
