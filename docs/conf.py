"""Sphinx configuration for the cup1d documentation."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).parent / "_build/matplotlib")
)

project = "cup1d"
author = "cup1d developers"
try:
    release = importlib.metadata.version("cup1d")
except importlib.metadata.PackageNotFoundError:
    release = "development"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Keep API discovery usable in documentation-only environments where the
# scientific stack or the external LaCE package is unavailable.
autodoc_mock_imports = [
    "camb",
    "lace",
    "mpi4py",
]

html_theme = (
    "pydata_sphinx_theme"
    if importlib.util.find_spec("pydata_sphinx_theme") is not None
    else "alabaster"
)
html_title = f"cup1d {release}"
html_theme_options = (
    {"show_toc_level": 2, "navigation_with_keys": True}
    if html_theme == "pydata_sphinx_theme"
    else {}
)

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = {}
if os.environ.get("CUP1D_DOCS_INTERSPHINX") == "1":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
