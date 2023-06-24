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

Default Values
^^^^^^^^^^^^^^

If the user is not using the ``pipeline`` and is not at the ``BSUO`` location, then the user has the option to change those initial global settings with this function:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 189-217

Where the default value is displayed but the user can change them to whatever value they like. There are no ``try excepts`` to catch incorrect ``types`` so the user has to be extra careful when entering in values.

Reduction Functions
^^^^^^^^^^^^^^^^^^^

Main functions of this program are ``bias``, ``dark``, ``flat``, and ``science``. Howeverm they all call this function for the actual data reducing of each image:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 220-289

Each process of reducing bias, darks, flats, and science image has its own section within the if statement.

Bias
^^^^

The bias function as shown below, first takes a flat image from the raw folder (when not using the pipeline) and displays it for the user to see:

+ If there is an overscan region
+ Where the data section (trim region) is

The format for both of these variables is [columns, rows] as is the fits convention. However, if the user does not want to use all of the columns like for an overscan region, then an example would be this ``[2073:2115, :]``. Where the only columns used are between 2073 and 2115 but all of the rows are being used. Likewise, an example trim region would be ``[20:2060, 12:2057]``. This example uses columns 20-2060 and rows 12-2057.

For ``ccdproc``, there is a weird bug where if the user enters the same rows for an overscan region and trim region, ``ccdproc`` errors out. So the recommendation is to use all of the rows for an overscan region and then specify where the user wants to trim the image after.

Once the each image has been reduced, they must be combined to create what is called a ``master bias`` or ``zero`` image. This is done specifically by these lines:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 354-360

The ``sigma_clip_dev_func`` computes the standard deviation about the center value (see `here <https://ccdproc.readthedocs.io/en/stable/api/ccdproc.Combiner.html#ccdproc.Combiner.sigma_clipping>`_ for more details regarding this). Also, the ``mem_limit`` can be changed but is set toa  default value of 16 Gb (1600e6). This might need to be reduced to be between 6 and 10 Gb as the average RAM is now 16 Gb. The purpose of this value is to aim to reduce and split the task of combining into chunks to reduce system RAM usage.

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 292-370

Dark
^^^^

Once the ``zero`` image has been created, the next step of the process is to create a ``master_dark``. However, some researchers forgo taking dark images as CCD's are cooled down so much that the thermal noise created by them in darks is virtually negligible and this is why there is a variable called ``dark_bool`` which has a default value of ``True``.

The process is the exact same as above for Bias except, the ``zero`` image is subtracted off each and every dark image. The combining of these newly reduced darks is also slightly different:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 393-441

As we now have to take into consideration the ``read nooise`` and ``gain`` of the CCD/camera.

Flat
^^^^

As stated above, the process is virtually the same but now the ``zero`` and ``master dark`` are both subtracted off each and every flat image. Now the combining of the images is slightly different as we now have filters.

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 477-487

This created ``master flats`` in each filter that the user is using.

Science
^^^^^^^

Again as stated above, the ``zero`` and ``master dark`` are subtracted from each science image and the ``master flats`` are used based on the filter used for each science image.

Adding to the Header
^^^^^^^^^^^^^^^^^^^^

Each of the previous four functions discussed all call the ``add_header`` function within the reduction loops. This function adds various values to the headers of each individual image:

.. literalinclude:: ../EclipsingBinaries/IRAF_Reduction.py
   :lines: 531-568

The goal of this is to make it easier in the future to tell what values were used in the reduction process.

BJD_TDB
^^^^^^^

The conversion between ``HJD`` and ``BJD_TDB`` is not an easy conversion. The purpose of including this in this package is to have a single time value across multiple telescopes or satellites. `TESS <https://tess.mit.edu/>`_ uses ``BJD_TDB`` while BSUO and various `SARA <https://www.saraobservatory.org/>`_ use ``HJD``.
