Reporting Issues and Contributing Code
======================================

Reporting Issues
----------------

If you have found a bug in EclipsingBinaries, please report it by creating a new issue on the `EclipsingBinaries issue tracker <https://github.com/kjkoeller/EclipsingBinaries/issues>`_.

Please include an example that demonstrates the issue and will allow for developers to reproduce and fix the problem. Please also provide information regarding your operating system and Python version.

Contributing Code
-----------------

We accept contributions at all levels, spanning from fixing a simple typo to developing major features. We welcome contributors who will abide by the `EclipsingBinaries Code of Conduct <https://github.com/kjkoeller/EclipsingBinaries/blob/main/CODE_OF_CONDUCT.md>`_.

EclipsingBinaries uses multiple workflows and coding guidelines to make sure that any code contributed will run across multiple Python versions and multiple operating systems.

Development Setup
-----------------

Follow the development installation in :doc:`installation`, then before opening a pull request:

+ Run ``pytest``. The suite runs offline, and new behaviour should come with tests.
+ Build the docs with ``sphinx-build -b html docs docs/_build/html`` if you changed them, and
  update the pages for any program whose inputs or outputs you changed.
+ Lint settings for ``flake8`` and ``pycodestyle`` live in ``tox.ini``.
+ Package metadata and dependencies live in ``pyproject.toml``. There is no ``setup.cfg``.
+ ``CHANGELOG.md`` is generated from merged pull requests by ``auto-changelog`` when a
  release is made, so it doesn't need editing by hand.

Some older source files use Windows (CRLF) line endings. Keep a file's existing line
endings when editing it so diffs stay readable.
