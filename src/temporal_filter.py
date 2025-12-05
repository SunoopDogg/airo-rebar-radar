"""Temporal filtering module using Kalman Filter for track stabilization."""

from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter

from .utils.config import KalmanFilterConfig


@dataclass
class Track:
    """Represents a tracked rebar across frames."""
    track_id: int
    center_x: float
    center_y: float
    radius: float
    confidence: float
    age: int = 0  # Number of frames since creation
    hits: int = 1  # Number of successful associations
    misses: int = 0  # Number of consecutive misses
    kalman_filter: KalmanFilter | None = field(default=None, repr=False)


class TemporalFilter:
    """Temporal filtering and track management using Kalman Filter."""

    def __init__(self, config: KalmanFilterConfig | None = None):
        """Initialize temporal filter.

        Args:
            config: Kalman filter configuration
        """
        self.config = config or KalmanFilterConfig()
        self.tracks: list[Track] = []
        self._next_track_id = 0
        self._max_misses = 3  # Maximum consecutive misses before track deletion

    def _create_kalman_filter(
        self,
        initial_x: float,
        initial_y: float,
        initial_r: float
    ) -> KalmanFilter:
        """Create a Kalman filter for tracking circle parameters.

        State: [x, y, r, vx, vy]
        Measurement: [x, y, r]

        Args:
            initial_x: Initial center x
            initial_y: Initial center y
            initial_r: Initial radius

        Returns:
            Configured KalmanFilter
        """
        kf = KalmanFilter(dim_x=5, dim_z=3)

        # State transition matrix (constant velocity model)
        dt = 1.0  # Time step between frames
        kf.F = np.array([
            [1, 0, 0, dt, 0],  # x = x + vx*dt
            [0, 1, 0, 0, dt],  # y = y + vy*dt
            [0, 0, 1, 0, 0],   # r = r (constant)
            [0, 0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 0, 1],   # vy = vy
        ])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])

        # Initial state
        kf.x = np.array([initial_x, initial_y, initial_r, 0, 0])

        # Initial covariance
        kf.P *= 0.1

        # Process noise
        q = self.config.process_noise
        kf.Q = np.diag([q, q, q * 0.1, q, q])

        # Measurement noise
        r = self.config.measurement_noise
        kf.R = np.diag([r, r, r * 0.5])

        return kf

    def _compute_distance(
        self,
        track: Track,
        detection: tuple[float, float, float]
    ) -> float:
        """Compute distance between track prediction and detection.

        Args:
            track: Existing track
            detection: Tuple of (x, y, r)

        Returns:
            Euclidean distance between predicted and detected positions
        """
        det_x, det_y, _ = detection
        return np.sqrt((track.center_x - det_x)**2 + (track.center_y - det_y)**2)

    def _associate_detections(
        self,
        detections: list[tuple[float, float, float, float]]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections with existing tracks using Hungarian algorithm.

        Simple greedy association based on distance threshold.

        Args:
            detections: List of (x, y, r, confidence) tuples

        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
            - matches: List of (track_idx, detection_idx) pairs
            - unmatched_tracks: List of track indices without matches
            - unmatched_detections: List of detection indices without matches
        """
        if not self.tracks or not detections:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))
            return [], unmatched_tracks, unmatched_detections

        # Compute distance matrix
        n_tracks = len(self.tracks)
        n_detections = len(detections)
        dist_matrix = np.full((n_tracks, n_detections), np.inf)

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                dist = self._compute_distance(track, det[:3])
                if dist <= self.config.max_distance:
                    dist_matrix[i, j] = dist

        # Greedy matching
        matches = []
        matched_tracks = set()
        matched_detections = set()

        while True:
            # Find minimum distance
            min_val = np.min(dist_matrix)
            if min_val == np.inf:
                break

            # Get indices
            track_idx, det_idx = np.unravel_index(
                np.argmin(dist_matrix), dist_matrix.shape
            )

            matches.append((track_idx, det_idx))
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)

            # Mark row and column as used
            dist_matrix[track_idx, :] = np.inf
            dist_matrix[:, det_idx] = np.inf

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_detections = [i for i in range(n_detections) if i not in matched_detections]

        return matches, unmatched_tracks, unmatched_detections

    def update(
        self,
        detections: list[tuple[float, float, float, float]]
    ) -> list[Track]:
        """Update tracks with new detections.

        Args:
            detections: List of (x, y, r, confidence) tuples

        Returns:
            List of active tracks after update
        """
        # Predict step for all tracks
        for track in self.tracks:
            if track.kalman_filter is not None:
                track.kalman_filter.predict()
                track.center_x = float(track.kalman_filter.x[0])
                track.center_y = float(track.kalman_filter.x[1])
                track.radius = float(track.kalman_filter.x[2])

        # Associate detections with tracks
        matches, unmatched_tracks, unmatched_detections = \
            self._associate_detections(detections)

        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det_x, det_y, det_r, det_conf = detections[det_idx]

            if track.kalman_filter is not None:
                # Kalman update
                track.kalman_filter.update(np.array([det_x, det_y, det_r]))
                track.center_x = float(track.kalman_filter.x[0])
                track.center_y = float(track.kalman_filter.x[1])
                track.radius = float(track.kalman_filter.x[2])
            else:
                # Direct update
                track.center_x = det_x
                track.center_y = det_y
                track.radius = det_r

            # Update confidence with exponential moving average
            track.confidence = 0.7 * track.confidence + 0.3 * det_conf
            track.hits += 1
            track.misses = 0
            track.age += 1

        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.misses += 1
            track.age += 1
            track.confidence *= 0.9  # Decay confidence

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det_x, det_y, det_r, det_conf = detections[det_idx]
            new_track = Track(
                track_id=self._next_track_id,
                center_x=det_x,
                center_y=det_y,
                radius=det_r,
                confidence=det_conf,
                kalman_filter=self._create_kalman_filter(det_x, det_y, det_r)
            )
            self.tracks.append(new_track)
            self._next_track_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses < self._max_misses]

        return self.tracks

    def get_stable_tracks(self, min_hits: int = 2) -> list[Track]:
        """Get tracks that have been consistently detected.

        Args:
            min_hits: Minimum number of hits required

        Returns:
            List of stable tracks
        """
        return [t for t in self.tracks if t.hits >= min_hits]

    def reset(self):
        """Reset all tracks."""
        self.tracks = []
        self._next_track_id = 0
