.. _toolbox:

Program Options
===============

Menu
----

After running the main script as described `here <https://eclipsingbinaries.readthedocs.io/en/latest/EB.html>`_. The user will be tasked with selecting different programs to run which is run by the ``menu.py`` file.

The various programs to choose from are:

+ IRAF Reduction
+ Find Minimum
+ TESS Database Search/Download
+ AIJ Comparison Star Selector
+ BSUO or SARA/TESS Night Filters
+ O-C Plotting
+ Gaia Search
+ O'Connell Effect
+ Color Light Curve
+ Close Program

IRAF Reduction
--------------

Making heavy use of `Astropy's ccdproc <https://ccdproc.readthedocs.io/en/stable/ccddata.html>`_ and `Photutils <https://photutils.readthedocs.io/en/stable/aperture.html>`_ I was able to create an automatic data reduction process using Bias, Darks, and Flats to reduce science images.

The first thing that is done is set some global variables:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 32-40

If the user is using the ``pipeline`` as described in this `section <https://eclipsingbinaries.readthedocs.io/en/latest/pipeline.html>`_ then the following lines of code are used:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 110-141

Otherwise, these lines are used from the ``main`` function:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 55-108

The only functional difference in these lines of code, is with the ``pipeline`` the user does not have to enter in folder and file locations manually where not using the ``pipeline`` the user does.
