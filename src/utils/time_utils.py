"""Utilities for time formatting and duration tracking."""


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable H:MM:SS string.

    Args:
        seconds: Time duration in seconds.

    Returns:
        str: Formatted duration string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
