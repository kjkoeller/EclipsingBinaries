# Configuration file for the Sphinx documentation builder.

import datetime
import os
import sys

# Get configuration information from setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
conf = ConfigParser()

conf.read([os.path.join(os.path.dirname(__file__), '..', 'setup.cfg')])
setup_cfg = dict(conf.items('metadata'))

# By default, highlight as Python 3.
highlight_language = 'python3'

# exclude_patterns.append('_templates')

# -- Project information

project = setup_cfg['name']
author = setup_cfg['author']
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, setup_cfg['author'])

__import__(project)
package = sys.modules[project]

ver = package.__version__
version = '.'.join(ver.split('.'))[:5]
release = ver

html_title = '{0} v{1}'.format(project, release)

# The master toctree document.
root_doc = 'index'

man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]

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

github_issues_url = 'https://github.com/astropy/ccdproc/issues/'
nitpicky = True

