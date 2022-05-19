import sys
import logging
import random
import argparse
from uuid import uuid4

from typing import Dict, Any, Optional

import numpy as np

import torch


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_id(cfg: Dict[str, Any]) -> None:
    """Set the 'id' field in the configuration."""
    cfg["id"] = uuid4().hex


def log(msg: str, path_to_log: Optional[str] = None) -> None:
    """
    Log a message to stdout and optionally also a log file.

    Args:
        msg: Message to be logged.
        path_to_log: optional, if given message will also be appended to file.
    """
    print(msg)
    if path_to_log is not None:
        with open(path_to_log, "a") as f:
            f.write(msg + "\n")


def init_logger(
    path_to_log: Optional[str] = None,
    level: Optional[int] = logging.DEBUG,
    format_str: Optional[str] = "[%(asctime)s|%(levelname)s] %(message)s",
) -> logging.Logger:
    """
    Initialize logger to stdout and optionally also a log file.

    Args:
        path_to_log: optional, if given message will also be appended to file.
        level: optional, logging level, see Python's logging library.
        format_str: optional, logging format string, see Python's logging library.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(format_str)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if path_to_log is not None:
        file_handler = logging.FileHandler(path_to_log)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def init_argument_parser() -> argparse.ArgumentParser:
    """Initialize basic CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="../data", help="Parent directory for datasets.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="../checkpoints",
        help="Parent directory for model states, results and logging.",
    )
    parser.add_argument(
        "--id", type=str, default=None, help="Unique experiment ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Training flag, if set models will be retrained, not loaded.",
    )
    return parser
