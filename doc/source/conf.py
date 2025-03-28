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
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.mathmpl",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

plot_include_source = True
plot_html_show_source_link = False
plot_rcparams = {"figure.dpi": 200, "savefig.dpi": 200}
