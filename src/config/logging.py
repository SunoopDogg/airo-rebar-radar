"""Logging configuration for rebar detection system."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance for the module
    """
    return logging.getLogger(f"rebar_radar.{name}")
