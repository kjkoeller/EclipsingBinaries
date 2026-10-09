.. _EB:

Running EclipsingBinaries
=========================

Starting the App
----------------

After installing (see :doc:`installation`), start the GUI from any terminal, Anaconda
prompt or command line with::

  EclipsingBinaries

The left side lists the available programs, and the right side shows the inputs and output
log for whichever one is selected. Each program is described in :ref:`toolbox`.

Working in the GUI
------------------

+ **Placeholder text** in a field (shown in grey) is only an example and is never used as a
  value. Fields that have a real default say so in the docs for that program, and use it
  when left empty.
+ **Paths** can be typed, picked with the **Browse** button, or dragged onto the field from
  Finder or Explorer. Paths starting with ``~`` are expanded to your home folder.
+ **Run** starts the program in the background so the window stays responsive. While it
  runs, **Run** is greyed out and **Cancel** asks the program to stop at its next
  checkpoint. Only one program runs at a time.
+ **Output log** shows progress and any errors. Switching to another program while one is
  running is fine; its remaining messages go to the terminal you launched from.
+ **Closing** the window (or pressing Cmd-Q on macOS or Ctrl-Q elsewhere) asks for
  confirmation, and offers to cancel a running task first.

Platform Notes
--------------

+ **macOS** — About and Quit are in the application menu. The app keeps a light theme in
  Dark Mode so all text stays readable. When started from a terminal the menu bar shows
  "Python" rather than "EclipsingBinaries", since it isn't a bundled ``.app``.
+ **Windows** — the app is DPI-aware, so text stays sharp on high-resolution displays.
+ **Drag and drop** needs the ``tkinterdnd2`` package, which supports Tk 8.6 and Tk 9. If it
  can't load (for example on an Intel Mac running Tk 9) the app still starts, prints a note
  in the terminal, and the **Browse** buttons work as usual.
+ **Missing tkinter** — if starting the app fails with a message about tkinter, see the
  tkinter notes in :doc:`installation`.

Using the Programs from Python
------------------------------

Every program in the GUI is a regular function you can call from a script or notebook.
They all accept two optional arguments:

+ ``write_callback`` — a function that receives each log message. Without one, messages go
  to Python's ``logging`` module if you have set it up, and are printed otherwise.
+ ``cancel_event`` — a ``threading.Event``; setting it asks a long-running function to stop.

For example:

.. code-block:: python

    from EclipsingBinaries.OConnell import main as oconnell

    oconnell(filepath="results", filter_files=["B.txt", "V.txt"],
             obj_name="NSVS_896797", period=0.3175, hjd=2458403.58763)

For unattended runs at the telescope, see :ref:`pipeline`.
