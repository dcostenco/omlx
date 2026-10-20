# SPDX-License-Identifier: Apache-2.0
"""Run owned-control regressions against this checkout without loading MLX."""

import os
from pathlib import Path
import sys
import unittest


def main():
    directory = Path(__file__).resolve().parent
    os.environ["OMLX_CONTROL_ROOT"] = str(directory.parents[1])
    suite = unittest.defaultTestLoader.discover(
        str(directory), pattern="test_provider*.py"
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    # A green empty suite or silently skipped CPU proof is not acceptance.
    return int(not result.wasSuccessful() or not result.testsRun or bool(result.skipped))


if __name__ == "__main__":
    sys.exit(main())
