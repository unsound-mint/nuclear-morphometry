"""Console + per-run file logging (spec section 34)."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(run_dir: Path, *, level: int = logging.INFO) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "run.log")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
