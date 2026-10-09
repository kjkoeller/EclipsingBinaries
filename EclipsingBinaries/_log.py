"""
Shared message routing for the analysis modules.

Most functions take an optional write_callback so the GUI can show their
progress in its log panel. This picks where a message goes when there is no
callback: the logging module if the caller has set logging up (the pipeline
does, so messages land in its log file), and plain print otherwise so
scripts and notebooks still see output.
"""

import logging

_logger = logging.getLogger("EclipsingBinaries")


def make_logger(write_callback=None):
    """Return a function that takes one message string and sends it to the right place."""
    if write_callback is not None:
        return write_callback

    def log(message):
        if _logger.hasHandlers():
            _logger.info(message)
        else:
            print(message)

    return log
