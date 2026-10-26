"""Utility helpers: seed management, safe logging, timing, and JSON serialization."""
from __future__ import annotations
import json
import logging
import os
import random
import time
from typing import Any, Callable, Dict, Tuple
import numpy as np


def setup_logger(name: str = "fraud_detection", level: int = logging.INFO) -> logging.Logger:
    """Configure a structured, secure logger that never logs raw unmasked card numbers or PII."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to convert NumPy scalars, arrays, and Pandas types into native Python."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


def benchmark_latency(func: Callable, *args, iterations: int = 20, **kwargs) -> Tuple[Any, float]:
    """Measure the average per-call execution latency in milliseconds."""
    # Warm-up run
    res = func(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(iterations):
        func(*args, **kwargs)
    t1 = time.perf_counter()
    avg_latency_ms = ((t1 - t0) / iterations) * 1000.0
    return res, avg_latency_ms
