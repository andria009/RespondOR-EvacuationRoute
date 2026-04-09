"""
Shared logging configuration for RespondOR-EvacuationRoute.

All entry points (src/main.py, experiments/*) call setup_logging() once at
startup.  It configures:
  - Console handler  — INFO and above, human-readable timestamp
  - File handler     — DEBUG and above, written to logs/{name}_{timestamp}.log

Log files are kept in the top-level logs/ directory so they are easy to find
and are gitignored.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(
    name: str,
    level: str = "INFO",
    logs_dir: Path = LOGS_DIR,
) -> Path:
    """
    Configure root logger with a console handler and a file handler.

    Args:
        name:     Short identifier used in the log filename (e.g. "preview_region",
                  "main", "export_shp").  Timestamp is appended automatically.
        level:    Minimum log level for the console handler (default INFO).
                  The file handler always captures DEBUG and above.
        logs_dir: Directory for log files (default: logs/).

    Returns:
        Path to the log file that was opened.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{name}_{timestamp}.log"

    root = logging.getLogger()
    # Only configure once — guard against double-calls (e.g. module reloads)
    if root.handlers:
        return log_path

    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(f"Logging to {log_path}")
    return log_path
