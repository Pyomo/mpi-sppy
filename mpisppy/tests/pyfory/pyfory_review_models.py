"""Run the maintained repository-root pyfory model probe."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root_probe = Path(__file__).resolve().parents[3] / "pyfory_review_models.py"
    sys.path.insert(0, str(root_probe.parent))
    runpy.run_path(str(root_probe), run_name="__main__")
