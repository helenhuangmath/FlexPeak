"""Sphinx configuration for the FlexPeak documentation (Read the Docs)."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "FlexPeak"
author = "Hua Huang"
copyright = "2026, Hua Huang"
try:
    from flexpeak import __version__ as release
except Exception:  # documentation can still be built without the package's deps
    release = "1.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxarg.ext",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["pysam", "pyBigWig", "numba", "matplotlib", "pandas", "jinja2"]
autosummary_generate = False

html_theme = "sphinx_rtd_theme"
html_title = f"FlexPeak {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
}
html_context = {
    "display_github": True,
    "github_user": "helenhuangmath",
    "github_repo": "FlexPeak",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
