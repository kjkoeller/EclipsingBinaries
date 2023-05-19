# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'EclipsingBinaries'
copyright = '2023, Kyle Koeller'
author = 'Kyle Koeller'

release = '2.8'
version = '2.8.4a8'

# The master toctree document.
root_doc = 'index'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
