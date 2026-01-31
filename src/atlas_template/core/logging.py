"""
Logging Configuration

Structured logging setup optimized for containerized environments.
Outputs to stdout for Docker log aggregation compatibility.
"""

import sys
from logging.config import dictConfig

from atlas_template.core.config import settings


def setup_logging() -> None:
    """
    Initialize application logging with consistent formatting.

    Configuration:
        - Output: stdout (Docker/K8s friendly)
        - Format: Timestamp | Level | Module | Message
        - Level: Controlled via LOG_LEVEL env var

    Note:
        Call once at application startup (in lifespan handler).
    """
    log_level = settings.LOG_LEVEL.upper()

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,  # Preserve third-party loggers
        "formatters": {
            "default": {
                "format": log_format,
                "datefmt": date_format,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,  # stdout for Docker log drivers
                "formatter": "default",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "atlas_template": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,  # Prevent duplicate logs to root
            },
            "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",  # Suppress verbose SQL logs unless debugging
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }

    dictConfig(logging_config)
