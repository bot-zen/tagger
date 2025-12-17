import logging
from typing import Optional


DEFAULT_FMT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(level: str = "INFO", fmt: str = DEFAULT_FMT, datefmt: str = DEFAULT_DATEFMT) -> logging.Logger:
    """
    Configure root logging once and return the package logger.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=_normalize_level(level), format=fmt, datefmt=datefmt)
    else:
        root.setLevel(_normalize_level(level))
    return logging.getLogger("tagger")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _normalize_level(level: str) -> int:
    try:
        return logging.getLevelName(level.upper())
    except Exception:
        return logging.INFO
