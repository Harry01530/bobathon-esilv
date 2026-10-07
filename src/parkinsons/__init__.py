"""Package root for the Parkinson's UPDRS prediction project.

Exposes ``PROJECT_ROOT``: the absolute path to the project root,
derived from this file's location. Any module that needs to resolve
a project-relative path (data files, configs) imports this constant
instead of hard-coding a CWD-relative string.

The package is installed in editable mode, so ``__file__`` points back
into the source tree at ``<root>/src/parkinsons/__init__.py`` and
``parents[2]`` is the project root.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
