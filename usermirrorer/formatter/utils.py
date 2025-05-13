from datetime import datetime, timedelta

def process_timestamp(timestamp: float | str | datetime) -> datetime:
    """Process timestamp into datetime object based on MIND dataset format."""
    if isinstance(timestamp, float):
        return datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    elif isinstance(timestamp, datetime):
        return timestamp
    else:
        raise ValueError(f"Invalid timestamp type: {type(timestamp)}")

def format_time_diff(td: timedelta) -> str:
    """Format time difference into human readable string for MIND dataset."""
    seconds = int(td.total_seconds())

    # Define time units in seconds
    units = [
        ('year', 365 * 24 * 60 * 60),
        ('month', 30 * 24 * 60 * 60),
        ('day', 24 * 60 * 60),
        ('hour', 60 * 60),
        ('minute', 60)
    ]
    
    # Find largest appropriate unit
    for unit, unit_seconds in units:
        if seconds >= unit_seconds:
            value = seconds // unit_seconds
            return f"{value} {unit}{'s' if value != 1 else ''} ago"
    
    return "just now"

