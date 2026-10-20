# SPDX-License-Identifier: Apache-2.0
"""Run the CPU-only protocol suite in CI without leaking its import guard."""

from pathlib import Path
import subprocess
import sys


def test_owned_control_cpu_protocol():
    runner = Path(__file__).with_name("owned_control_cpu") / "run.py"
    result = subprocess.run(
        [sys.executable, "-I", str(runner)],
        cwd=runner.parents[2],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
