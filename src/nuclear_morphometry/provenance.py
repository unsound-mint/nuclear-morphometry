"""Run provenance (spec section 27.1).

Every run writes a provenance.json capturing enough to reproduce it months
later: code version, config/manifest hashes, package versions, and (when
available) GPU/Cellpose identity. Optional dependencies are probed via
``importlib.util.find_spec`` so this module works on a CPU-only machine.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nuclear_morphometry.config import Config

# Git state describes the code that ran, so it is read from the package's own
# source tree, not the process cwd (which is usually the data directory).
_PACKAGE_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    """Stdout of a git command run in the package directory, or None on any error."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_PACKAGE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _in_package_repo() -> bool:
    """True when the package sits in its own checkout, not in an unrelated enclosing repo
    (e.g. a non-editable install inside some project's .venv)."""
    top = _git("rev-parse", "--show-toplevel")
    return top is not None and (Path(top.strip()) / "src" / "nuclear_morphometry").is_dir()


def current_git_commit() -> str | None:
    if not _in_package_repo():
        return None
    out = _git("rev-parse", "HEAD")
    return out.strip() if out else None


def _git_dirty() -> bool | None:
    """Whether the package checkout has uncommitted changes; None when unknown."""
    if not _in_package_repo():
        return None
    out = _git("status", "--porcelain")
    return None if out is None else bool(out.strip())


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_provenance() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_runtime_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info


def cellpose_provenance() -> dict[str, Any]:
    version = package_version("cellpose")
    return {"available": version is not None, "version": version}


def build_provenance(
    *,
    config: Config,
    config_path: Path,
    config_hash: str,
    manifest_hash: str,
    run_id: str,
) -> dict[str, Any]:
    """Assemble the provenance record at run start. Callers add end_time later."""
    manifest_path = config.experiment.manifest
    manifest_stat = manifest_path.stat() if manifest_path.exists() else None

    return {
        "run_id": run_id,
        "start_time_utc": datetime.now(UTC).isoformat(),
        "end_time_utc": None,
        "git_commit": current_git_commit(),
        "git_dirty": _git_dirty(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            name: package_version(name)
            for name in (
                "nuclear-morphometry",
                "numpy",
                "scipy",
                "scikit-image",
                "polars",
                "pyarrow",
                "pydantic",
                "typer",
                "bioio",
                "bioio-czi",
                "bioio-tifffile",
                "tifffile",
            )
        },
        "torch": torch_provenance(),
        "cellpose": cellpose_provenance(),
        "segmentation_config": config.segmentation.model_dump(by_alias=True),
        "random_seeds": {"qc_random_seed": config.qc.random_seed},
        "manifest_file": {
            "size_bytes": manifest_stat.st_size if manifest_stat else None,
            "mtime_utc": (
                datetime.fromtimestamp(manifest_stat.st_mtime, tz=UTC).isoformat()
                if manifest_stat
                else None
            ),
        },
    }


def finalize_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Set end_time_utc on an existing provenance record. Mutates and returns it."""
    provenance["end_time_utc"] = datetime.now(UTC).isoformat()
    return provenance
