"""
A GUI to centralize all the scripts and capabilities of this package for ease of use for the user and
making it more convenient to use and access than a command line or individual scripts.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 05/21/2025
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import textwrap


def dynamic_import(progress_queue):
    packages = [
        ("tkinter", "from tkinter import messagebox, filedialog, ttk"),
        ("pathlib", "from pathlib import Path"),
        ("astropy", "from astropy.nddata import CCDData"),
        ("astroquery", "from astroquery.mast import Tesscut"),
        ("matplotlib", "from matplotlib import pyplot as plt"),
        ("json", "import json"),
        ("os", "import os"),
        ("traceback", "import traceback"),
        ("custom_scripts", textwrap.dedent("""
            from .IRAF_Reduction import run_reduction
            from .tess_data_search import run_tess_search
            from .apass import comparison_selector
            from .multi_aperture_photometry import main as multi_ap
            from .gaia import target_star as gaia
            from .OConnell import main as oconnell
            from .version import __version__
        """)),
    ]
    total = len(packages)

    try:
        for i, (name, command) in enumerate(packages, start=1):
            exec(command, globals())
            # print(f"DEBUG: {name} imported successfully")  # Debug statement
            progress_queue.put((i, total, f"Loading {name}..."))
            time.sleep(0.5)

        progress_queue.put((total, total, "Finalizing"))

    except Exception as e:
        progress_queue.put((None, None, f"Error loading {name}: {e}"))
        print(f"ERROR: {e}")  # Debug output for errors

    progress_queue.put(None)


class SplashScreen(tk.Toplevel):
    """
    Splash Screen with progress bar and dynamic loading messages.
    """
    def __init__(self, root, progress_queue, on_close_callback):
        super().__init__(root)
        self.progress_queue = progress_queue
        self.on_close_callback = on_close_callback
        self.configure(bg="#003366")

        # Center the splash screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        splash_width = int(screen_width * 0.4)
        splash_height = int(screen_height * 0.3)
        x_position = (screen_width - splash_width) // 2
        y_position = (screen_height - splash_height) // 2
        self.geometry(f"{splash_width}x{splash_height}+{x_position}+{y_position}")
        self.overrideredirect(True)  # Hide title bar

        # Title Label
        self.title_label = tk.Label(
            self,
            text="EclipsingBinaries",
            font=("Helvetica", int(splash_height * 0.1), "bold"),
            fg="white",
            bg="#003366"
        )
        self.title_label.pack(pady=int(splash_height * 0.05))

        # Dynamic Loading Label
        self.message_label = tk.Label(
            self,
            text="Initializing...",
            font=("Helvetica", int(splash_height * 0.05)),
            fg="white",
            bg="#003366"
        )
        self.message_label.pack(pady=int(splash_height * 0.05))

        # Progress Bar with style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", thickness=20, troughcolor="#002244", background="#00bfff")
        self.progress = ttk.Progressbar(
            self, orient="horizontal", length=int(splash_width * 0.8), mode="determinate", style="TProgressbar"
        )
        self.progress.pack(pady=int(splash_height * 0.1))

        # Start monitoring the queue
        self.after(100, self.monitor_progress)

    def monitor_progress(self):
        """
        Monitor the progress queue and update the splash screen.
        """
        try:
            progress_data = self.progress_queue.get_nowait()
            if progress_data is None:  # Should never receive a raw None due to safeguards
                self.on_close_callback()
                self.destroy()  # Close the splash screen when done
                return

            current, total, message = progress_data
            if current is not None and total is not None:  # Ensure valid data for progress
                self.progress["value"] = (current / total) * 100
                self.message_label.config(text=message)

            self.after(100, self.monitor_progress)  # Check again after 100 ms
        except queue.Empty:
            self.after(100, self.monitor_progress)  # If no data, keep checking


class ProgramLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        # Get screen dimensions
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # Set window size as 75%x75% of screen dimensions
        self.window_width = int(self.screen_width * 0.75)
        self.window_height = int(self.screen_height * 0.75)
        self.minsize(1000, 700)  # Minimum size for usability

        # Center the window
        self.center_window()

        # Window setup
        self.title("EclipsingBinaries")
        self.configure(bg="#f5f5f5")

        # Bind click event to remove focus
        self.bind("<Button-1>", self.remove_focus)

        # Configure rows and columns for resizing
        self.rowconfigure(0, weight=1)  # Header
        self.rowconfigure(1, weight=8)  # Main content
        self.rowconfigure(2, weight=1)  # Footer
        self.columnconfigure(0, weight=1)  # Full GUI width

        # Dynamic font scaling
        self.header_font = ("Helvetica", max(16, self.window_width // 50), "bold")
        self.label_font = ("Helvetica", max(10, self.window_width // 100))
        self.button_font = ("Helvetica", max(12, self.window_width // 80))

        # Create UI components
        self.create_header()
        self.create_layout()
        self.create_menu()

        # Initialize cancel_event and current_task
        self.cancel_event = threading.Event()
        self.current_task = None

        # Footer
        footer_label = tk.Label(
            self.left_frame,
            text="Developed by Kyle Koeller | Eclipsing Binaries Research",
            font=("Helvetica", 10),
            bg="#f5f5f5",
        )
        footer_label.pack(side="bottom", pady=5)

        # Bind quit to autosave settings
        self.protocol("WM_DELETE_WINDOW", self.quit_program)

    def create_input_field(
            self, parent, label_text, placeholder_text, row, variable=None, validation_func=None, error_message=""
    ):
        """
        Create a labeled input field with placeholder functionality, validation, and inline error message.
        """
        # Create a label for the input field
        tk.Label(parent, text=label_text, font=self.label_font, bg="#ffffff").grid(
            row=row, column=0, padx=10, pady=5, sticky="e"
        )

        # Create the entry widget
        entry = tk.Entry(parent, width=30, font=self.label_font)  # Reduced width
        entry.grid(row=row, column=1, padx=10, pady=5, sticky="w")

        # Error message label (initially empty)
        error_label = tk.Label(
            parent, text="", font=("Helvetica", 9), fg="red", bg="#ffffff"  # Smaller font size
        )
        error_label.place(in_=entry, relx=1.05, rely=0.5, anchor="w")  # Positioned to the right of the entry

        # Placeholder functionality
        def on_focus_in(event):
            if entry.get() == placeholder_text and entry["fg"] == "gray":
                entry.delete(0, "end")
                entry.config(fg="black")

        def on_focus_out(event):
            if not entry.get().strip():  # If entry is empty
                entry.insert(0, placeholder_text)
                entry.config(fg="gray")
            validate_input()  # Validate on focus out

        # Validation function
        def validate_input():
            value = entry.get().strip()
            if value == placeholder_text or not value:  # Input is empty
                error_label.config(text=error_message or "This field is required.")
                entry.config(bg="#ffe6e6")  # Highlight entry with a light red background
            elif validation_func and not validation_func(value):  # Validation fails
                error_label.config(text=error_message)
                entry.config(bg="#ffe6e6")
            else:  # Valid input
                error_label.config(text="")
                entry.config(bg="white")

        # Bind events
        entry.insert(0, placeholder_text)
        entry.config(fg="gray")
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

        # Bind variable (if provided)
        if variable:
            entry.config(textvariable=variable)

        return entry

    def remove_focus(self, event):
        """Remove focus from the currently focused widget unless it's an input widget."""
        widget = self.focus_get()
        clicked_widget = self.winfo_containing(event.x_root, event.y_root)

        # Reset focus only if the clicked widget is not an input widget
        if not isinstance(clicked_widget, (tk.Entry, tk.Text, ttk.Combobox)):
            self.focus_set()

    def center_window(self):
        """Center the window on the screen"""
        x_position = (self.screen_width - self.window_width) // 2
        y_position = (self.screen_height - self.window_height) // 2
        self.geometry(f"{self.window_width}x{self.window_height}+{x_position}+{y_position}")

    def create_header(self):
        """Create the header section"""
        header_frame = tk.Frame(self, bg="#003366")
        header_frame.pack(fill="x")

        tk.Label(header_frame, text="EclipsingBinaries", font=self.header_font, fg="white", bg="#003366").pack(pady=10)
        tk.Label(header_frame, text="Refer to the GitHub README for more details:", font=self.label_font, fg="white",
                 bg="#003366").pack()
        tk.Label(header_frame, text="https://github.com/kjkoeller/EclipsingBinaries/", font=self.label_font,
                 fg="#aadfff", bg="#003366").pack(pady=5)

    def create_layout(self):
        """Create the main layout with left and right frames"""
        self.left_frame = tk.Frame(self, bg="#f5f5f5")
        self.left_frame.place(relx=0, rely=0.2, relwidth=0.3, relheight=0.8)

        self.right_frame = tk.Frame(self, bg="#ffffff", relief="groove", bd=2)
        self.right_frame.place(relx=0.3, rely=0.2, relwidth=0.7, relheight=0.8)

    def create_menu(self):
        """Create the options menu with the Help functionality."""
        # Create the top menu bar
        menubar = tk.Menu(self)

        # File Menu (Placeholder for extensibility)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit_program)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help Contents", command=self.open_help_window)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

        # Left-side options menu (original buttons)
        options = [
            ("IRAF Reduction", self.show_iraf_reduction),
            ("Find Minimum (WIP)", self.dummy_action),
            ("TESS Database Search/Download", self.show_tess_search),
            ("AIJ Comparison Star Selector", self.show_aij_comparison_selector),
            ("Multi-Aperture Calculation", self.show_multi_aperture_photometry),
            ("BSUO or SARA/TESS Night Filters", self.dummy_action),
            ("O-C Plotting", self.dummy_action),
            ("Gaia Search", self.show_gaia_query),
            ("O'Connell Effect", self.show_oconnell_effect),
            ("Color Light Curve", self.dummy_action),
            ("Close Program", self.quit_program),
        ]

        for option, command in options:
            self.create_menu_button(option, command)

    def create_menu_button(self, text, command):
        """Create a menu button"""
        tk.Button(self.left_frame, text=text, command=command,
                  font=self.button_font,
                  bg="#003366", fg="white",
                  activebackground="#00509e", activeforeground="white",
                  relief="flat", cursor="hand2",
                  highlightbackground="#003366",  # Needed for macOS
                  highlightthickness=1).pack(pady=5, padx=10, fill="x")

    def open_help_window(self):
        """Open a separate Help window."""
        help_window = tk.Toplevel(self)
        help_window.title("Help Contents")
        help_window.geometry("600x400")
        help_window.configure(bg="#f5f5f5")

        help_content = (
            "Welcome to the Help Window!\n\n"
            "Here you can find instructions and tips on how to use the application.\n"
            "\nFeatures:\n"
            "- IRAF Reduction: Process raw astronomical images.\n"
            "- TESS Search: Retrieve TESS sector data.\n"
            "- AIJ Comparison: Select comparison stars.\n"
            "- O'Connell Effect: Calculate light curve effects.\n"
            "\nFor more information, visit the GitHub repository:\n"
            "https://github.com/kjkoeller/EclipsingBinaries/"
        )

        help_label = tk.Label(help_window, text=help_content, font=("Helvetica", 10), justify="left", wraplength=550)
        help_label.pack(pady=20, padx=20, anchor="w")

    def show_about(self):
        """Display an About dialog."""
        messagebox.showinfo(
            "About EclipsingBinaries",
            "EclipsingBinaries\n\n"
            f"Version: {__version__}\n"
            "Author: Kyle Koeller\n\n"
            "For support, visit the GitHub repository:\n"
            "https://github.com/kjkoeller/EclipsingBinaries/"
        )

    def cancel_task(self):
        """Cancel the currently running task."""
        if self.current_task and self.current_task.is_alive():
            if messagebox.askyesno("Cancel Task", "Are you sure you want to cancel the current task?"):
                self.write_to_log("Task cancellation requested...")
                self.cancel_event.set()
        else:
            messagebox.showinfo("No Task Running", "There is no task currently running.")

    def write_to_log(self, message):
        """Write a message to the log area and ensure it updates"""
        self.log_area.insert("end", message + "\n")
        self.log_area.see("end")  # Scroll to the latest message
        self.update()  # Process pending GUI events to refresh the log

    def clear_right_frame(self):
        """Clear the right frame by destroying all its widgets"""
        for widget in self.right_frame.winfo_children():
            widget.destroy()

    def create_checkbox(self, parent, text, variable, row):
        """Create a checkbox with alignment"""
        tk.Checkbutton(parent, text=text, variable=variable, font=self.label_font, bg="#ffffff").grid(
            row=row, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
        )

    def create_scrollbar_and_log(self, row):
        # Create a frame to hold the log area and scrollbar
        log_frame = tk.Frame(self.right_frame, bg="#ffffff")
        log_frame.grid(row=row, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # Add the Text widget (log area) and the Scrollbar
        self.log_area = tk.Text(log_frame, wrap="word", height=12, font=("Helvetica", 10))
        scrollbar = tk.Scrollbar(log_frame, command=self.log_area.yview)

        # Configure the Text widget to work with the scrollbar
        self.log_area.configure(yscrollcommand=scrollbar.set)

        # Pack the Text widget and the scrollbar
        self.log_area.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure grid for log area
        self.right_frame.grid_rowconfigure(row, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

    def create_run_button(self, parent, action, row, **kwargs):
        """Create the 'Run' button dynamically with proper alignment"""
        tk.Button(parent, text="Run", font=self.button_font,
              bg="#003366", fg="white",
              activebackground="#00509e", activeforeground="white",
              relief="flat", cursor="hand2",
              highlightbackground="#003366", highlightthickness=1,
              command=lambda: action(**kwargs)).grid(row=row, column=0, columnspan=2, pady=20)

    def create_cancel_button(self, parent, action, row, **kwargs):
        tk.Button(parent, text="Cancel", font=self.button_font,
              bg="#003366", fg="white",
              activebackground="#00509e", activeforeground="white",
              relief="flat", cursor="hand2",
              highlightbackground="#003366", highlightthickness=1,
              command=lambda: action(**kwargs)).grid(row=row, column=1, columnspan=2, pady=20)

    def run_task(self, target, *args):
        """Run a task in a separate thread with cancellation support."""
        self.cancel_event.clear()  # Reset cancel_event for the new task
        self.current_task = threading.Thread(target=target, args=args, daemon=True)
        self.current_task.start()

    def show_iraf_reduction(self):
        """Display the IRAF reduction panel."""
        self.clear_right_frame()

        # Configure grid for centering
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add a title
        tk.Label(self.right_frame, text="IRAF Reduction", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Input fields with placeholders
        raw_images_path = self.create_input_field(self.right_frame, "Raw Images Path:",
                                                  "C:\\folder1\\raw_images", row=1,
                                                  validation_func=lambda x: len(x.strip()) > 0, # Empty string
                                                  error_message="File path cannot be empty.")

        calibrated_images_path = self.create_input_field(self.right_frame, "Calibrated Images Path:",
                                                         "C:\\folder1\\calibrated_images", row=2,
                                                         validation_func=lambda x: len(x.strip()) > 0, # Empty string
                                                         error_message="File path cannot be empty.")

        location = self.create_input_field(self.right_frame, "Location:",
                                           "e.g., BSUO, CTIO, etc.", row=3,
                                           validation_func=lambda x: isinstance(x, str) and x.isalpha(), # String
                                           error_message="Value must be a string.")

        # Checkbox for dark frames
        dark_bool_var = tk.BooleanVar(value=True)
        self.create_checkbox(self.right_frame, "Use Dark Frames", dark_bool_var, row=4)

        # Overscan and Trim region inputs
        overscan = self.create_input_field(self.right_frame, "Overscan Region:",
                                           "[2073:2115, :]", row=5,
                                           validation_func=lambda x: len(x.strip()) > 0, # Empty
                                           error_message="Please enter at least [:,:].")

        trim = self.create_input_field(self.right_frame, "Trim Region:",
                                       "[20:2060, 12:2057]", row=6,
                                       validation_func=lambda x: len(x.strip()) > 0, # Empty
                                       error_message="Please enter at least [:,:].")

        # Button to open and plot bias image
        tk.Button(self.right_frame, text="Open Bias Image", font=self.button_font, bg="#003366", fg="white",
                  command=self.open_bias_image).grid(row=7, column=0, columnspan=2, pady=10, sticky="")

        # Run button
        self.create_run_button(self.right_frame, self.run_iraf_reduction, row=8,
                               raw_images_path=raw_images_path,
                               calibrated_images_path=calibrated_images_path,
                               location=location,
                               dark_bool_var=dark_bool_var,
                               overscan_var=overscan,
                               trim_var=trim
                               )

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=9, column=0, columnspan=2, pady=5
        )

        # Create the scroll bar and log area
        self.create_scrollbar_and_log(10)

    def open_bias_image(self):
        """Open a bias image, plot it, and prompt for regions."""
        file_path = filedialog.askopenfilename(title="Select Bias Image", filetypes=[("FITS files", "*.fits"),
                                                                                     ("FITS files", "*.fit"),
                                                                                     ("FITS files", "*.fts")])
        if file_path:
            try:
                # Read the FITS file as CCDData
                ccd = CCDData.read(file_path, unit="adu")
                self.bias_plot(ccd)

                # Log success
                self.write_to_log(f"Successfully loaded and plotted bias image: {file_path}")
            except Exception as e:
                self.write_to_log(f"Failed to load bias image: {e}")

    def bias_plot(self, ccd):
        """Plot the count values for row 1000 to determine overscan and trim regions."""
        plt.figure(figsize=(10, 5))
        plt.plot(ccd.data[1000][:], label="Raw Bias")
        plt.grid()
        plt.axvline(x=2077, color="black", linewidth=2, linestyle="dashed", label="Suggested Start of Overscan")
        plt.legend()
        plt.xlim(-50, ccd.data.shape[1] + 50)
        plt.xlabel("Pixel Number")
        plt.ylabel("Counts")
        plt.title("Bias Image Analysis: Row 1000")
        plt.show()

    def show_tess_search(self):
        """Display the TESS Database Search panel."""
        self.clear_right_frame()

        # Configure grid
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add title
        tk.Label(self.right_frame, text="TESS Database Search", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Input fields
        system_name = self.create_input_field(self.right_frame, "System Name:",
                                              "NSVS 896797", row=1,
                                              validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                              error_message="Please enter a system name.")
        download_path = self.create_input_field(self.right_frame, "Download Path:",
                                                "C:\\folder1\\download", row=2,
                                                validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                                error_message="Please enter a file pathway.")

        # Checkbox for download specific sector
        download_all_var = tk.BooleanVar(value=False)  # Default to unchecked

        # Create placeholder for dynamic widgets
        self.sector_dropdown = None
        self.retrieve_button = None
        self.sector_label = None

        tk.Checkbutton(
            self.right_frame, text="Download Specific Sector", variable=download_all_var,
            font=self.label_font, bg="#ffffff",
            command=lambda: self.toggle_sector_options(download_all_var, system_name)
        ).grid(row=3, column=0, columnspan=2, pady=5)

        # Run button
        self.create_run_button(self.right_frame, self.run_tess_search, row=6,
                               system_name=system_name,
                               download_path=download_path,
                               download_all_var=download_all_var)

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=7, column=0, columnspan=2, pady=5
        )

        # Create scrollbar for the log area
        self.create_scrollbar_and_log(8)

    def retrieve_sectors(self, system_name, sector_dropdown):
        """Retrieve available sectors for a given TIC ID."""
        try:
            system_name_value = system_name.get().strip()
            if not system_name_value:
                self.write_to_log("Error: System name (TIC ID) is required.")
                return

            # Simulate a search and retrieve sectors
            self.write_to_log(f"Retrieving sectors for: {system_name_value}")
            sector_table = Tesscut.get_sectors(objectname=system_name_value)

            if not sector_table:
                self.write_to_log(f"No TESS data found for system {system_name_value}.")
                return

            # Log and populate dropdown
            formatted_table = "\n".join(sector_table.pformat(show_name=True, max_width=-1, align="^"))
            self.write_to_log("Available Sectors:\n" + formatted_table)

            self.available_sectors = list(sector_table["sector"])
            sector_dropdown["values"] = self.available_sectors
            sector_dropdown.set("Select a Sector")
            self.write_to_log("Sectors successfully retrieved.")
        except Exception as e:
            self.write_to_log(f"Error retrieving sectors: {e}")

    def toggle_sector_options(self, download_all_var, system_name):
        """Show or hide the sector dropdown, 'Select Specific Sector' label, and Retrieve Sectors button."""
        if download_all_var.get():  # Checkbox is selected (True)
            # Add "Select Specific Sector" label
            if not self.sector_label:
                self.sector_label = tk.Label(self.right_frame, text="Select Specific Sector:", font=self.label_font,
                                             bg="#ffffff")
                self.sector_label.grid(row=5, column=0, sticky="e")

            # Add sector dropdown
            if not self.sector_dropdown:
                self.sector_dropdown = ttk.Combobox(self.right_frame, state="readonly", values=[], font=self.label_font)
                self.sector_dropdown.grid(row=5, column=1, padx=10, pady=5, sticky="w")

            # Add "Retrieve Sectors" button
            if not self.retrieve_button:
                self.retrieve_button = tk.Button(
                    self.right_frame, text="Retrieve Sectors", font=self.button_font, bg="#003366", fg="white",
                    command=lambda: self.retrieve_sectors(system_name=system_name, sector_dropdown=self.sector_dropdown)
                )
                self.retrieve_button.grid(row=4, column=0, columnspan=2, pady=10)
        else:  # Checkbox is unselected (False)
            # Remove "Select Specific Sector" label
            if self.sector_label:
                self.sector_label.destroy()
                self.sector_label = None

            # Remove sector dropdown
            if self.sector_dropdown:
                self.sector_dropdown.destroy()
                self.sector_dropdown = None

            # Remove "Retrieve Sectors" button
            if self.retrieve_button:
                self.retrieve_button.destroy()
                self.retrieve_button = None

    def show_aij_comparison_selector(self):
        """Display the AIJ Comparison Star Selector panel."""
        self.clear_right_frame()

        # Configure grid for centering
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add title
        tk.Label(self.right_frame, text="AIJ Comparison Star Selector", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Input fields
        ra = self.create_input_field(self.right_frame, "Right Ascension (RA):",
                                     "HH:MM:SS.SSSS", row=1,
                                     validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                     error_message="Please enter a RA.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                      error_message="Please enter a DEC.")

        folder_path = self.create_input_field(self.right_frame, "Data Save Folder Path:",
                                              "C:\\folder1\\download", row=3,
                                              validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                              error_message="Please enter a file pathway.")

        obj_name = self.create_input_field(self.right_frame, "Object Name:",
                                           "NSVS 896797", row=4,
                                           validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                           error_message="Please enter the object name.")

        science_image = self.create_input_field(self.right_frame, "Science Image Folder Path:",
                                                "C:\\folder1\\calibrated_images", row=5,
                                                validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                                error_message="Please enter a file pathway.")

        # Buttons for comparison selector
        tk.Button(self.right_frame, text="Run Comparison Selector", font=self.button_font, bg="#003366", fg="white",
                  command=lambda: self.run_comparison_selector(ra, dec, folder_path, obj_name, science_image)).grid(
            row=6, column=0, columnspan=2, pady=10, sticky=""
        )

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=7, column=0, columnspan=2, pady=5
        )

        # Create scrollbar for the log area
        self.create_scrollbar_and_log(8)

    def show_multi_aperture_photometry(self):
        """Display the Multi-Aperture Photometry panel."""
        self.clear_right_frame()

        # Configure grid
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add title
        tk.Label(self.right_frame, text="Multi-Aperture Photometry", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Input fields
        obj_name = self.create_input_field(self.right_frame, "Object Name:",
                                           "NSVS 896797", row=1,
                                           validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                           error_message="Please enter an object name.")

        reduced_images_path = self.create_input_field(self.right_frame, "Reduced Images Path:",
                                                      "C:\\folder1\\reduced_images", row=2,
                                                      validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                                      error_message="Please enter a file pathway.")

        radec_b_file = self.create_input_field(self.right_frame, "RADEC File (B Filter):",
                                               "C:\\folder1\\B.radec", row=3,
                                               validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                               error_message="Please enter a file pathway with file name")

        radec_v_file = self.create_input_field(self.right_frame, "RADEC File (V Filter):",
                                               "C:\\folder1\\V.radec", row=4,
                                               validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                               error_message="Please enter a file pathway with file name.")

        radec_r_file = self.create_input_field(self.right_frame, "RADEC File (R Filter):",
                                               "C:\\folder1\\R.radec", row=5,
                                               validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                               error_message="Please enter a file pathway with file name.")

        # Run button
        self.create_run_button(self.right_frame, self.run_multi_aperture_photometry, row=6,
                               obj_name=obj_name,
                               reduced_images_path=reduced_images_path,
                               radec_b_file=radec_b_file,
                               radec_v_file=radec_v_file,
                               radec_r_file=radec_r_file)

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=7, column=0, columnspan=2, pady=5
        )

        # Create scrollbar for the log area
        self.create_scrollbar_and_log(8)

    def show_gaia_query(self):
        """Display the Gaia query panel."""
        self.clear_right_frame()

        # Configure grid for centering
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add title
        tk.Label(self.right_frame, text="Gaia Query", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Input fields
        ra = self.create_input_field(self.right_frame, "Right Ascension (RA):",
                                     "HH:MM:SS.SSSS", row=1,
                                     validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                     error_message="Please enter a RA.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                      error_message="Please enter a DEC.")

        output_file = self.create_input_field(self.right_frame, "Output File Path:",
                                              "C:\\folder1\\Gaia_[star name].txt", row=3,
                                              validation_func=lambda x: len(x.strip()) > 0,  # Empty
                                              error_message="Please enter a file pathway.")

        # Run button
        self.create_run_button(self.right_frame, self.run_gaia_query, row=4,
                               ra=ra,
                               dec=dec,
                               output_file=output_file)

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=5, column=0, columnspan=2, pady=5
        )

        # Create scrollbar for the log area
        self.create_scrollbar_and_log(6)

    def show_oconnell_effect(self):
        """Display the O'Connell Effect panel."""
        self.clear_right_frame()

        # Configure grid
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add title
        tk.Label(self.right_frame, text="O'Connell Effect Calculation", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Variable for filter count
        filter_count_var = tk.IntVar(value=3)  # Default selection is 3 filters

        # File path variables
        file_path_vars = [tk.StringVar() for _ in range(3)]

        # Function to dynamically display entry fields
        def update_file_path_fields():
            for i in range(3):
                if i < filter_count_var.get():
                    file_entries[i].grid()
                else:
                    file_entries[i].grid_remove()

        # Radio buttons for selecting filter count
        tk.Label(self.right_frame, text="Select Number of Filters:", font=self.label_font, bg="#ffffff").grid(
            row=1, column=0, columnspan=2, pady=5, sticky=""
        )

        filter_frame = tk.Frame(self.right_frame, bg="#ffffff")
        filter_frame.grid(row=2, column=0, columnspan=2, sticky="")

        for i in range(1, 4):
            tk.Radiobutton(
                filter_frame,
                text=f"{i} Filter{'s' if i > 1 else ''}",
                variable=filter_count_var,
                value=i,
                command=update_file_path_fields,
                font=self.label_font,
                bg="#ffffff",
                anchor="w"
            ).pack(side="left", padx=10)

        # Create entry fields for file paths
        file_entries = []
        filter_list = ["B", "V", "R"]
        for i in range(3):
            entry = self.create_input_field(
                parent=self.right_frame,
                label_text=f"File Path {i + 1}:",
                placeholder_text=f"C:\\folder1\\{filter_list[i]}.txt",
                row=3 + i,
                variable=file_path_vars[i]
            )
            file_entries.append(entry)

        # Initially hide extra file entries
        update_file_path_fields()

        # HJD input
        hjd_var = self.create_input_field(
            parent=self.right_frame,
            label_text="HJD:",
            placeholder_text="2458403.58763",
            row=6,
            validation_func=lambda x: x.replace(".", "", 1).isdigit(),  # Is a float
            error_message="Please enter a valid HJD."
        )

        # Period input
        period_var = self.create_input_field(self.right_frame, "Period:",
                                             "0.3175", row=7,
                                             validation_func=lambda x: x.replace(".", "", 1).isdigit(),  # Is a float
                                             error_message="Please enter a valid Period.")

        obj_name_var = self.create_input_field(self.right_frame, "System Name:",
                                               "NSVS_896797", row=8,
                                               validation_func=lambda x: len(x.strip()) > 0,  # Empty,  # Is a float
                                               error_message="Please enter a System Name."
                                               )

        # Output file path
        output_var = self.create_input_field(self.right_frame, "Output File Path:",
                                             "C:/folder1/folder2", row=9,
                                             validation_func=lambda x: len(x.strip()) > 0,  # Empty,  # Is a float
                                             error_message="Please enter a file pathway."
                                             )

        # Button to run O'Connell Effect calculation
        tk.Button(
            self.right_frame,
            text="Run O'Connell Effect",
            font=self.button_font,
            bg="#003366",
            fg="white",
            command=lambda: self.run_oconnell_effect(
                filter_count_var,
                file_entries,
                hjd_var,
                period_var,
                obj_name_var,
                output_var
            )
        ).grid(row=10, column=0, columnspan=2, pady=20)

        # Create scrollbar for the log area
        self.create_scrollbar_and_log(11)

    def run_iraf_reduction(self, raw_images_path, calibrated_images_path, location, dark_bool_var, overscan_var, trim_var):
        """Run the IRAF reduction process in a separate thread"""

        def reduction_task():
            try:
                self.create_cancel_button(self.right_frame, self.cancel_task, row=8)

                raw_path = Path(raw_images_path.get().strip())
                calibrated_path = Path(calibrated_images_path.get().strip())
                loc = location.get().strip()
                use_dark_frames = dark_bool_var.get()
                trim_region = trim_var.get().strip()
                overscan_region = overscan_var.get().strip()

                # Validate paths
                if not raw_path.exists():
                    self.write_to_log("Error: Raw images path does not exist.")
                    raise FileNotFoundError("Invalid raw images path.")
                if not calibrated_path.exists():
                    self.write_to_log("Error: Calibrated images path does not exist.")
                    raise FileNotFoundError("Invalid calibrated images path.")

                # Log setup
                self.write_to_log("Paths validated successfully.")
                self.write_to_log(f"Raw Images Path: {raw_path}")
                self.write_to_log(f"Calibrated Images Path: {calibrated_path}")
                self.write_to_log(f"Location: {loc}")
                self.write_to_log(f"Use Dark Frames: {'Yes' if use_dark_frames else 'No'}")
                self.write_to_log("Starting IRAF Reduction...\n")

                # Call the IRAF Reduction script
                run_reduction(
                    path=raw_path,
                    calibrated=calibrated_path,
                    location=loc,
                    cancel_event=self.cancel_event,  # Pass cancel_event
                    dark_bool=use_dark_frames,
                    overscan_region=overscan_region,
                    trim_region=trim_region,
                    write_callback=self.write_to_log
                )

                if not self.cancel_event.is_set():
                    # self.write_to_log("IRAF Reduction completed successfully!")
                    messagebox.showinfo("Success", "IRAF Reduction completed successfully!")
                else:
                    messagebox.showinfo("Cancelled", "IRAF Reduction was cancelled.")

            except Exception as e:
                self.write_to_log(f"An error occurred during IRAF Reduction: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())
                # messagebox.showerror("Error", f"An error occurred during IRAF Reduction: {e}")

        # Run the reduction in a separate thread
        self.run_task(reduction_task)

    def run_tess_search(self, system_name, download_path, download_all_var):
        """Run the TESS Database Search and TESSCut processing in a separate thread."""

        def search_task():
            try:
                self.create_cancel_button(self.right_frame, self.cancel_task, row=6)

                system_name_value = system_name.get().strip()
                download_path_value = download_path.get().strip()
                download_all = download_all_var.get()

                specific_sector_value = None
                if download_all:  # Specific sector mode
                    if self.sector_dropdown and self.sector_dropdown.get().isdigit():
                        specific_sector_value = int(self.sector_dropdown.get())
                    else:
                        self.write_to_log("Error: Please select a valid sector.")
                        return

                # Validate inputs
                if not system_name_value:
                    self.write_to_log("Error: System name (TIC ID) is required.")
                    return
                if not download_path_value:
                    self.write_to_log("Error: Download path is required.")
                    return

                # Log setup
                self.write_to_log(f"System Name: {system_name_value}")
                self.write_to_log(f"Download Path: {download_path_value}")
                self.write_to_log(f"Download All Sectors: {'Yes' if download_all else 'No'}")
                if specific_sector_value is not None:
                    self.write_to_log(f"Specific Sector: {specific_sector_value}")

                # Run TESS search
                run_tess_search(
                    system_name=system_name_value,
                    download_all=download_all,
                    specific_sector=specific_sector_value,
                    download_path=download_path_value,
                    write_callback=self.write_to_log,
                    cancel_event=self.cancel_event  # Pass cancel_event
                )

                if not self.cancel_event.is_set():
                    messagebox.showinfo("Success", "TESS Database Search completed successfully.")
                else:
                    messagebox.showinfo("Cancelled", "TESS Database Search was canceled.")

            except Exception as e:
                self.write_to_log(f"An error occurred during TESS database search: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(search_task)

    def run_comparison_selector(self, ra, dec, folder_path, obj_name, science_image):
        """Run the comparison selector in a separate thread."""

        def selector_task():
            try:
                self.create_cancel_button(self.right_frame, self.cancel_task, row=6)

                # Retrieve input values
                ra_value = ra.get().strip()
                dec_value = dec.get().strip()
                folder_value = folder_path.get().strip()
                obj_value = obj_name.get().strip()
                science_image_value = science_image.get().strip()

                if not all([ra_value, dec_value, folder_value, obj_value, science_image_value]):
                    self.write_to_log("Error: All fields are required.")
                    return

                self.write_to_log(f"Running comparison selector for object: {obj_value}")
                comparison_selector(ra=ra_value,
                                    dec=dec_value,
                                    pipeline=False,
                                    folder_path=folder_value,
                                    obj_name=obj_value,
                                    science_image=science_image_value,
                                    write_callback=self.write_to_log,
                                    cancel_event=self.cancel_event  # Pass cancel_event
                                    )

                if not self.cancel_event.is_set():
                    self.write_to_log("Comparison Selector completed successfully.")
                else:
                    self.write_to_log("Comparison Selector was canceled.")
            except Exception as e:
                self.write_to_log(f"An error occurred during comparison star selection: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(selector_task)

    def run_multi_aperture_photometry(self, obj_name, reduced_images_path, radec_b_file, radec_v_file, radec_r_file):
        """Run the Multi-Aperture Photometry script in a separate thread."""

        def photometry_task():
            try:
                self.create_cancel_button(self.right_frame, self.cancel_task, row=6)

                obj_name_value = obj_name.get().strip()
                reduced_path_value = reduced_images_path.get().strip()
                radec_b_path = radec_b_file.get().strip()
                radec_v_path = radec_v_file.get().strip()
                radec_r_path = radec_r_file.get().strip()

                # Validate inputs
                if not reduced_path_value:
                    self.write_to_log("Error: Reduced images path is required.")
                    return
                if not (radec_b_path and radec_v_path and radec_r_path):
                    self.write_to_log("Error: RADEC files for all filters are required.")
                    return

                # Log setup
                self.write_to_log(f"Object Name: {obj_name_value}")
                self.write_to_log(f"Reduced Images Path: {reduced_path_value}")
                self.write_to_log(f"RADEC File (B Filter): {radec_b_path}")
                self.write_to_log(f"RADEC File (V Filter): {radec_v_path}")
                self.write_to_log(f"RADEC File (R Filter): {radec_r_path}")

                # Run the multi-aperture photometry script
                multi_ap(
                    path=reduced_path_value,
                    pipeline=False,
                    radec_list=[radec_b_path, radec_v_path, radec_r_path],
                    obj_name=obj_name_value,
                    write_callback=self.write_to_log,
                    cancel_event=self.cancel_event  # Pass cancel_event
                )

                if not self.cancel_event.is_set():
                    self.write_to_log("Multi-Aperture Photometry completed successfully.")
                else:
                    self.write_to_log("Multi-Aperture Photometry was canceled.")
            except Exception as e:
                self.write_to_log(f"An error occurred during Multi-Aperture Photometry: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        # Run the photometry task in a separate thread
        self.run_task(photometry_task)

    def run_gaia_query(self, ra, dec, output_file):
        """Run the Gaia query in a separate thread."""

        def gaia_query():
            try:
                self.create_cancel_button(self.right_frame, self.cancel_task, row=4)

                ra_value = ra.get().strip()
                dec_value = dec.get().strip()
                output_path = output_file.get().strip()

                if not all([ra_value, dec_value, output_path]):
                    self.write_to_log("Error: All fields are required.")
                    return

                self.write_to_log(f"Running Gaia Query for RA: {ra_value}, DEC: {dec_value}")

                gaia(
                    ra_input=ra_value,
                    dec_input=dec_value,
                    output_path=output_path,
                    write_callback=self.write_to_log,
                    cancel_event=self.cancel_event  # Pass cancel_event

                )

            except Exception as e:
                self.write_to_log(f"An error occurred during Multi-Aperture Photometry: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(gaia_query)

    def run_oconnell_effect(self, filter_count, file_path_vars, hjd, period, obj_name, output_file):
        """
        Run the O'Connell Effect calculation based on the selected number of filters
        and provided input values.
        """

        def oconnell_task():
            try:
                filter_count_var = filter_count.get()
                # Collect file paths based on the selected filter count
                file_paths = [var.get().strip() for var in file_path_vars[:filter_count_var]]
                missing_paths = [path for path in file_paths if not path]

                hjd_value = hjd.get().strip()
                period_value = period.get().strip()
                obj_name_value = obj_name.get().strip()
                output_file_value = output_file.get().strip()

                # Validate inputs
                if missing_paths:
                    self.write_to_log("Error: Missing file paths for selected filters.")
                    messagebox.showerror("Input Error", "Please provide file paths for all selected filters.")
                    return
                for file_path in file_paths:
                    if not Path(file_path).exists():
                        self.write_to_log(f"Error: File does not exist - {file_path}")
                        messagebox.showerror("File Error", f"The file {file_path} does not exist.")
                        return
                if not hjd_value:
                    self.write_to_log("Error: HJD is required.")
                    messagebox.showerror("Input Error", "Please enter the Heliocentric Julian Date (HJD).")
                    return
                try:
                    hjd_value = float(hjd_value)
                except ValueError:
                    self.write_to_log(f"Error: Invalid HJD value - {hjd_value}")
                    messagebox.showerror("Input Error", f"Invalid HJD value: {hjd_value}")
                    return
                if not period_value:
                    self.write_to_log("Error: Period is required.")
                    messagebox.showerror("Input Error", "Please enter the period of the system.")
                    return
                if not obj_name_value:
                    self.write_to_log("Error: System Name is required.")
                    messagebox.showerror("Input Error", "Please enter the name of the system.")
                if not output_file_value:
                    self.write_to_log("Error: Output file path is required.")
                    messagebox.showerror("Input Error", "Please provide an output file path.")
                    return

                # Log inputs
                self.write_to_log("Starting O'Connell Effect calculation with the following inputs:")
                self.write_to_log(f"Number of Filters: {filter_count}")
                self.write_to_log(f"File Paths: {', '.join(file_paths)}")
                self.write_to_log(f"HJD: {hjd_value}")
                self.write_to_log(f"Period: {period_value}")
                self.write_to_log(f"Output File: {output_file_value}")

                # Run the O'Connell Effect calculation
                # from oconnell_effect import main as oconnell_main
                oconnell(
                    filepath=output_file_value,
                    filter_files=list(file_paths),
                    obj_name="OConnell_Output",
                    period=float(period_value),
                    hjd=float(hjd_value),
                    write_callback=self.write_to_log,
                    cancel_event=self.cancel_event  # Pass cancel_event
                )

                # Notify success
                self.write_to_log("O'Connell Effect calculation completed successfully.")
                messagebox.showinfo("Success", "O'Connell Effect calculation completed successfully!")

            except Exception as e:
                # Log and notify errors
                self.write_to_log(f"An error occurred during O'Connell Effect calculation: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        # Run the task in a separate thread
        self.run_task(oconnell_task)

    def dummy_action(self):
        """Dummy action for unimplemented features"""
        messagebox.showinfo("Action", "This feature is not implemented yet.")

    def quit_program(self):
        """Quit the program with confirmation and save settings."""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.destroy()

# class PlaceholderEntry(tk.Entry):
#     def __init__(self, master=None, placeholder="Enter text...", placeholder_color="grey", *args, **kwargs):
#         super().__init__(master, *args, **kwargs)
#
#         self.placeholder = placeholder
#         self.placeholder_color = placeholder_color
#         self.default_fg_color = self["fg"]
#
#         self.bind("<FocusIn>", self._clear_placeholder)
#         self.bind("<FocusOut>", self._add_placeholder)
#
#         self._add_placeholder()
#
#     def _clear_placeholder(self, event=None):
#         if self["fg"] == self.placeholder_color:
#             self.delete(0, tk.END)
#             self["fg"] = self.default_fg_color
#
#     def _add_placeholder(self, event=None):
#         if not self.get():
#             self.insert(0, self.placeholder)
#             self["fg"] = self.placeholder_color


def launch_main_gui():
    """
    Launch the main GUI after the splash screen.
    """
    app = ProgramLauncher()
    app.mainloop()


def main():
    # Create a hidden root window for the splash screen
    root = tk.Tk()
    root.withdraw()

    # Create a queue for progress updates
    progress_queue = queue.Queue()

    def start_main_gui():
        root.quit()  # Exit the splash screen's event loop

    # Initialize the splash screen
    SplashScreen(root, progress_queue, on_close_callback=start_main_gui)

    # Start a thread to dynamically import modules
    threading.Thread(target=dynamic_import, args=(progress_queue,), daemon=True).start()

    # Run the splash screen's event loop
    root.mainloop()

    # Once the splash screen closes, launch the main GUI
    root.destroy()  # Clean up the splash screen's hidden root window
    launch_main_gui()


if __name__ == "__main__":
    main()
