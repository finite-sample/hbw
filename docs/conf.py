"""Sphinx configuration for hbw documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

project = "hbw"
copyright = "2024, Gaurav Sood"
author = "Gaurav Sood"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

html_theme = "furo"
html_title = "hbw"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
