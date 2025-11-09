import logging
import sys


# logging_utils.py
import logging
import sys


# logger.py
import logging
from typing import Optional

# Keep track of the current "active" experiment log file
_current_log_file = None

def setup_logger(log_file: Optional[str] = None, level=logging.INFO):
    """
    Set up the root logger for the current experiment.
    All modules will get their loggers from this setup.
    """
    global _current_log_file
    _current_log_file = log_file

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File handler if a file is specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

def get_logger(name: str):
    """
    Get a logger for any module. Uses the root logger configured
    by `setup_logger`. Returns a standard logging.Logger.
    """

    # call setup_logger if it hasn't been called yet
    if not logging.getLogger().hasHandlers():
        setup_logger()
    return logging.getLogger(name)

