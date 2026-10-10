.. _EB:

Running EclipsingBinaries
=========================

Starting the App
----------------

After installing (see :doc:`installation`), start the GUI from any terminal, Anaconda
prompt or command line with::

  EclipsingBinaries

The sidebar on the left lists the programs grouped in the order you usually work through
them (Reduce, Comparison stars, Photometry, Analysis, Archives), with a gold bar marking the
open one. The right side shows the inputs and output log for that program. The app reopens
on whichever program you used last. Each program is described in :ref:`toolbox`.

Working in the GUI
------------------

+ **Placeholder text** in a field (shown in grey) is only an example and is never used as a
  value. Fields that have a real default say so in the docs for that program, and use it
  when left empty.
+ **Tooltips** — rest the pointer on a field's name for a short note on what it expects.
+ **Paths** can be typed, picked with the **Browse** button, or dragged onto the field from
  Finder or Explorer. Paths starting with ``~`` are expanded to your home folder.
+ **Remembered settings** — each program fills its fields with what you entered last time,
  including the filter choices and check boxes. They're kept in
  ``~/.EclipsingBinaries/gui_settings.json``. **File > Clear Saved Settings...** forgets them
  all. To keep a separate set (for example on a shared lab computer), point the
  ``ECLIPSINGBINARIES_SETTINGS`` environment variable at another file before starting the app.
+ **Run** starts the program in the background so the window stays responsive. While it
  runs, **Run** is greyed out and **Cancel** asks the program to stop at its next
  checkpoint. Only one program runs at a time.
+ **Status bar** — the bar along the bottom of the window shows which program is running,
  how long it has been going, and a moving progress bar. When it ends it says whether the
  program finished, was canceled, or stopped with an error, and how long it took.
+ **Results** — when a program finishes, buttons appear next to **Run**: **Open Output
  Folder** opens the folder in Finder, Explorer or your file manager, and most programs add
  a button for their main result, such as a plot or the light curves. These replace the
  pop-up messages earlier versions showed. The buttons per program are listed in
  :ref:`toolbox`.
+ **Output log** shows progress, with errors in red, warnings in amber and finished steps in
  green. **Save Log...** writes it to a text file (useful when reporting a problem) and
  **Clear** empties it. Switching to another program while one is running is fine; its
  remaining messages go to the terminal you launched from.
+ **Keyboard shortcuts** — Enter runs the open program (or presses the focused button),
  Escape offers to cancel a running program, and Cmd-Q on macOS or Ctrl-Q elsewhere quits.
+ **Help** — **Help > Help Contents** gives a short guide, and **Help > Documentation** and
  **Help > GitHub Repository** open this site and the source code.
+ **Closing** the window asks for confirmation, and offers to cancel a running task first.
  The open program's fields are saved when you close.

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
