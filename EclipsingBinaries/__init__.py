# __init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("EclipsingBinaries")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source without install)
    __version__ = "unknown"
