"""Test package bootstrap.

Ensures the repository root is importable as a package source so ``import
potato`` works whether or not the project has been installed (``pip install
-e .``). CI installs the package; running the suite locally without installing
still works because we prepend the repo root to ``sys.path`` here.

Standard library only (docs/rules.md: no dependency without approval).
"""

import pathlib
import sys

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
