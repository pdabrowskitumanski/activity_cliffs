"""
Utility functions for activity cliffs analysis.
"""

import json
import warnings
from pathlib import Path
import numpy as np
from rich import get_console
from rich.panel import Panel
from rich.text import Text


def _short_location(filename: str) -> str:
    """Shorten a warning's source path: package-relative for installed libraries, else relative to the project."""
    path = Path(filename)
    if "site-packages" in path.parts:
        return "/".join(path.parts[path.parts.index("site-packages") + 1:])
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parent.parent))
    except ValueError:
        return str(path)


def install_rich_warnings() -> None:
    """Render Python warnings as compact yellow rich panels instead of raw stderr text."""

    def _show_warning(message, category, filename, lineno, file=None, line=None):
        # Print through rich's global console, which also drives the progress bars,
        # so panels are placed above a running progress display instead of breaking it
        get_console().print(Panel(
            Text(str(message).strip()),
            title=f"[bold]⚠ {category.__name__}[/bold]",
            title_align="left",
            subtitle=f"[dim]{_short_location(filename)}:{lineno}[/dim]",
            subtitle_align="right",
            border_style="yellow",
            expand=False,
        ))

    warnings.showwarning = _show_warning


def load_config(config_name: str) -> dict:
    """
    Load a configuration from a JSON file in the configs/ directory.
    
    Args:
        config_name: Name of the config file (without .json extension).
    
    Returns:
        Dictionary with configuration parameters.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_path = Path("configs") / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def convert_to_native_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    
    Converts:
    - NumPy integers (int64, int32, etc.) to Python int
    - NumPy floats (float64, float32, etc.) to Python float
    - NumPy arrays to Python lists
    - Nested dictionaries and lists are processed recursively
    
    Args:
        obj: Object that may contain NumPy types.
    
    Returns:
        Object with all NumPy types converted to native Python types.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    return obj

