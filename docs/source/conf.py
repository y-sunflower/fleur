import os
import sys

sys.path.insert(0, os.path.abspath("./"))

project = "inferplot"
copyright = "2025, Joseph Barbier"
author = "Joseph Barbier"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

autoapi_dirs = ["../../inferplot"]
autoapi_ignore = ["*/__init__.py"]
autoapi_add_toctree_entry = True
