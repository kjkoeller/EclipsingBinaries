# Configuration file for the Sphinx documentation builder.

import datetime
import tomllib
from importlib.metadata import version as get_version
from unittest.mock import MagicMock
import os
import sys

# Mock heavy dependencies that are not needed to build docs
# and may not be installable in the ReadTheDocs environment
MOCK_MODULES = [
    'tkinterdnd2',
    'ccdproc',
    'photutils',
    'photutils.aperture',
    'pyia',
    'astroquery',
    'astroquery.mast',
    'PyAstronomy',
    'tqdm',
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# Project metadata lives in pyproject.toml
with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), 'rb') as f:
    project_meta = tomllib.load(f)['project']

# By default, highlight as Python 3.
highlight_language = 'python3'

# -- Project information
project = project_meta['name']
author = project_meta['authors'][0]['name']
copyright = '{0}, {1}'.format(datetime.datetime.now().year, author)

try:
    release = get_version("EclipsingBinaries")
except Exception:
    release = "unknown"
version = '.'.join(release.split('.')[:2])

html_title = '{0} v{1}'.format(project, release)

# -- General configuration
root_doc = 'index'

man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'myst_parser',  # renders CHANGELOG.md on the changelog page
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

templates_path = ['_templates']

# -- Options for HTML output
html_theme = "furo"
