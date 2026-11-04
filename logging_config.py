
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when Streamlit reruns the module.
    if not getattr(root, "_tripmind_logging_configured", False):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)

        file_handler = RotatingFileHandler(
            log_dir / "tripmind.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        root.addHandler(console)
        root.addHandler(file_handler)
        setattr(root, "_tripmind_logging_configured", True)

    return logging.getLogger("tripmind")


logger = configure_logging()
