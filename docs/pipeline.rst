Pipeline
========

Overview
--------

The pipeline automates the full data reduction workflow for eclipsing binary observations.
Once started, it monitors a directory for incoming images from the telescope, waits until
the observation session ends (no new files for a configurable timeout), then automatically
runs data reduction, comparison star selection, and multi-aperture photometry in sequence.

The pipeline also provides:

- **Process locking** — prevents two instances from running on the same directory simultaneously
- **Summary report** — writes a ``[object_name]_pipeline_summary.txt`` file to the output
  folder at the end of every run, recording the runtime and status of each stage

Usage
-----

To see all available options::

    EB_pipeline -h

To start the pipeline::

    EB_pipeline INPUT_DIR OUTPUT_DIR --ra HH:MM:SS.SS --dec DD:MM:SS.SS [options]

Required Inputs
---------------

``INPUT_DIR``
    Path to the folder where incoming raw images from the telescope will appear.
    The pipeline monitors this folder for new files.

``OUTPUT_DIR``
    Path to the folder where reduced images and all pipeline output files will be saved.

``--ra``
    Right ascension of the target in the format ``HH:MM:SS.SS``.

``--dec``
    Declination of the target in the format ``DD:MM:SS.SS``.
    For negative declinations use ``-DD:MM:SS.SS``.

Optional Inputs
---------------

``--time``
    How long in seconds the pipeline waits after the last new file before starting
    data reduction. Default is ``3600`` seconds (1 hour).

``--loc``
    Location of the telescope. Accepted values are ``BSUO`` or any site listed in the
    `Astropy sites list <https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json>`_.
    Default is ``None``.

``--name``
    Name of the target object. Use underscores instead of spaces (e.g. ``NSVS_254037``).
    This name is used for all output file names. Default is ``NSVS_254037``.

``--mem``
    Memory limit for the IRAF reduction stage in bytes. Default is ``450e6`` (450 MB).
    For example, to allow 800 MB use ``800e6``.

``--gain``
    Gain of the camera in electrons per ADU. Default is ``1.43``.

``--rdnoise``
    Readout noise of the camera in electrons. Default is ``10.83``.

``--log-file``
    Optional path to write all log output to a file in addition to the terminal.
    Useful for keeping a record of the night's run.

Output Files
------------

The pipeline writes the following files to ``OUTPUT_DIR`` at the end of a successful run:

- Reduced FITS images from the IRAF reduction stage
- RADEC comparison star files for each filter
- Light curve CSV files and plots for each filter
- ``[name]_pipeline_summary.txt`` — a summary of the run including total runtime,
  per-stage duration, and any warnings encountered

The summary report looks like this::

    ============================================================
    Pipeline Summary — NSVS_254037
    ============================================================
    Total runtime : 1:23:47

    Stages:
      IRAF Reduction                 0:45:12      [ok]
      Comparison Star Selection      0:02:15      [ok]
      Aperture Photometry            0:36:20      [ok]

    No warnings.
    ============================================================

Process Locking
---------------

The pipeline creates a ``.pipeline.lock`` file in ``OUTPUT_DIR`` when it starts.
This prevents a second instance from accidentally running on the same directory at
the same time. If the pipeline exits unexpectedly, the lock is released automatically.
If you are certain no other instance is running and the lock file remains, delete it
manually and restart.

Example
-------

A typical pipeline invocation::

    EB_pipeline C:/folder1/raw_images C:/folder1/reduced_images \
        --ra 00:28:27.96 --dec 78:57:42.65 \
        --name NSVS_254037 \
        --time 3000 \
        --loc CTIO \
        --gain 1.43 \
        --rdnoise 10.83 \
        --log-file C:/folder1/pipeline_run.log

For a negative declination::

    EB_pipeline C:/folder1/raw_images C:/folder1/reduced_images \
        --ra 00:28:27.96 --dec -12:34:56.78 \
        --name NSVS_254037

.. note::
    The ``INPUT_DIR`` and ``OUTPUT_DIR`` positional arguments must come first, in that
    order, before any ``--`` options. The ``--ra`` and ``--dec`` arguments are required
    and have no default values.

.. warning::
    Do not use the same folder for both ``INPUT_DIR`` and ``OUTPUT_DIR``. The reduction
    stage writes files to ``OUTPUT_DIR`` and monitoring ``INPUT_DIR`` for new files could
    behave unexpectedly if they overlap.
