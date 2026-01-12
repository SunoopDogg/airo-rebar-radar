"""Geometry utility functions for coordinate transformations and distance calculations."""

import math

import numpy as np


def rotate_and_translate(
    dx: float,
    dy: float,
    center_x: float,
    center_y: float,
    yaw: float,
) -> tuple[float, float]:
    """Apply rotation around origin then translate to center position.

    Performs 2D rotation by yaw angle (counter-clockwise positive) followed by
    translation to the specified center point.

    Args:
        dx: X offset from center (before rotation)
        dy: Y offset from center (before rotation)
        center_x: Center X coordinate for translation
        center_y: Center Y coordinate for translation
        yaw: Rotation angle in radians (positive = counter-clockwise)

    Returns:
        Tuple of (x, y) coordinates after rotation and translation
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x = center_x + dx * cos_yaw - dy * sin_yaw
    y = center_y + dx * sin_yaw + dy * cos_yaw
    return x, y


def rotate_points(
    points: list[tuple[float, float]],
    center_x: float,
    center_y: float,
    yaw: float,
) -> list[tuple[float, float]]:
    """Rotate and translate multiple points.

    Args:
        points: List of (dx, dy) offsets from center
        center_x: Center X coordinate for translation
        center_y: Center Y coordinate for translation
        yaw: Rotation angle in radians

    Returns:
        List of (x, y) coordinates after rotation and translation
    """
    return [
        rotate_and_translate(dx, dy, center_x, center_y, yaw)
        for dx, dy in points
    ]


def calculate_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Calculate Euclidean distance between two 2D points.

    Args:
        x1: X coordinate of first point
        y1: Y coordinate of first point
        x2: X coordinate of second point
        y2: Y coordinate of second point

    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_distance_np(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Calculate Euclidean distance using numpy (for consistency with numpy code).

    Args:
        x1: X coordinate of first point
        y1: Y coordinate of first point
        x2: X coordinate of second point
        y2: Y coordinate of second point

    Returns:
        Euclidean distance between the points
    """
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


