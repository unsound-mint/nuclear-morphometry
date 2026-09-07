"""Regression: dayana_nuclei.cli must not import napari/Qt at module load
time (AGENTS.md: "the computational core must not depend on napari or
Qt"). napari is the optional 'gui' dependency group -- if cli.py imported
qc/viewer.py (or napari itself) unconditionally, every `dayana-nuclei`
invocation would require it, not just `dayana-nuclei qc`.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_cli_does_not_import_napari() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dayana_nuclei.cli; "
            "assert 'napari' not in sys.modules, sorted(m for m in sys.modules if 'napari' in m)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
