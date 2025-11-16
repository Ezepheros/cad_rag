# logger.py
import logging
from typing import Optional
import threading
import time
import os

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


_current_log_file = None
_resource_monitor_thread = None
_stop_monitor = False

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


def log_resource_usage(logger: logging.Logger):
    """
    Log a one-time snapshot of CPU, RAM, and GPU usage.
    Call this inside training loops, evaluation steps,
    or anywhere you want a resource report.
    """
    process = psutil.Process(os.getpid())

    # CPU / RAM
    cpu = process.cpu_percent(interval=None)  # instant
    ram_mb = process.memory_info().rss / 1024**2
    sys_ram = psutil.virtual_memory().percent

    # GPU
    if _NVML_AVAILABLE:
        gpu_stats = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpu_stats.append(f"GPU{i}: {mem.used//(1024**2)}MB")
        gpu_str = ", ".join(gpu_stats)
    else:
        gpu_str = "No GPU (NVML not available)"

    logger.info(
        f"CPU: {cpu:.1f}% | RAM: {ram_mb:.1f}MB (system {sys_ram}%) | {gpu_str}"
    )

# Background Resource Monitor
def _resource_monitor_loop(interval: float):
    logger = logging.getLogger("ResourceMonitor")
    process = psutil.Process(os.getpid())

    while not _stop_monitor:
        try:
            # CPU / RAM
            cpu = process.cpu_percent(interval=None)
            ram_mb = process.memory_info().rss / (1024 ** 2)
            sys_ram = psutil.virtual_memory().percent

            # GPU
            gpu_str = "GPU: n/a"
            if _NVML_AVAILABLE:
                stats = []
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    stats.append(f"{mem.used // (1024**2)}MB")
                gpu_str = "GPU: " + ", ".join(stats)

            logger.info(f"CPU: {cpu:.1f}% | RAM: {ram_mb:.1f}MB (system {sys_ram}%) | {gpu_str}")

        except Exception as e:
            logger.warning(f"Resource monitor error: {e}")

        time.sleep(interval)


def start_resource_monitor(interval: float = 5.0):
    """Start background resource usage logging."""
    global _resource_monitor_thread, _stop_monitor

    if _resource_monitor_thread is not None:
        return  # already running

    _stop_monitor = False
    _resource_monitor_thread = threading.Thread(
        target=_resource_monitor_loop,
        args=(interval,),
        daemon=True,
    )
    _resource_monitor_thread.start()


def stop_resource_monitor():
    """Stop the background monitor."""
    global _stop_monitor
    _stop_monitor = True