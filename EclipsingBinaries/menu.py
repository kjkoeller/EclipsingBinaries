"""
A GUI to centralize all the scripts and capabilities of this package for ease of use for the user and
making it more convenient to use and access than a command line or individual scripts.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 04/21/2026
"""

import json
import platform
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from importlib.metadata import version, PackageNotFoundError
from tkinterdnd2 import TkinterDnD, DND_FILES
from astropy.nddata import CCDData
from astroquery.mast import Tesscut
from matplotlib import pyplot as plt

from .IRAF_Reduction import run_reduction
from .tess_data_search import run_tess_search
from .apass import comparison_selector
from .multi_aperture_photometry import main as multi_ap
from .gaia import target_star as gaia
from .OConnell import main as oconnell
from .color_light_curve import color_plot

try:
    __version__ = version("EclipsingBinaries")
except PackageNotFoundError:
    __version__ = "unknown"

# tk.Buttons don't style on macOS
if platform.system() == "Darwin":
    import tkmacosx as tkm
    Button = tkm.Button
else:
    Button = tk.Button


# class ProgramLauncher(tk.Tk):
class ProgramLauncher(TkinterDnD.Tk):
    """
    Main GUI that the user interacts with for each of the different options for analysis
    """

    def __init__(self):
        super().__init__()
        self.update_idletasks()  # Force full tkinter init before any method calls

        # Initialize ttk.Style
        self.style = ttk.Style()
        self.style.theme_use("default")  # Use a customizable ttk theme

        self.config_file = "config.json"

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

    def _make_validator(self, entry, error_label, error_message, validation_func):
        """Build and return the validate_input function for an entry field."""
        def validate_input():
            value = entry.get().strip()
            is_placeholder = entry["fg"] == "gray"  # Check if it's still styled as a placeholder
            if is_placeholder or not value:  # Input is empty
                error_label.config(text=error_message or "This field is required.")
                entry.config(bg="#ffe6e6")  # Highlight entry with a light red background
            elif validation_func and not validation_func(value):  # Validation fails
                error_label.config(text=error_message)
                entry.config(bg="#ffe6e6")
            else:  # Valid input
                error_label.config(text="")
                entry.config(bg="white")
        return validate_input

    def _make_placeholder_bindings(self, entry, placeholder_text, validate_input):
        """Bind focus-in, focus-out, and placeholder behaviour to an entry field."""
        def on_focus_in(event):
            if entry.get() == placeholder_text and entry["fg"] == "gray":
                entry.delete(0, "end")
                entry.config(fg="black")

        def on_focus_out(event):
            if not entry.get().strip():  # If entry is empty
                entry.insert(0, placeholder_text)
                entry.config(fg="gray")
            if not getattr(entry, '_browsing', False):  # Skip validation if browse dialog is open
                validate_input()  # Validate on focus out

        entry.insert(0, placeholder_text)
        entry.config(fg="gray")
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    def _make_browse_button(self, entry_frame, entry, browse_type, validate_input):
        """Add a browse button inside the entry frame if browse_type is specified."""
        if not browse_type:
            return

        def browse_action():
            selected_path = None
            if browse_type == "file":
                selected_path = filedialog.askopenfilename(title="Select File")
            elif browse_type == "folder":
                selected_path = filedialog.askdirectory(title="Select Folder")
            entry._browsing = False  # Reset flag after dialog closes
            if selected_path:
                entry.delete(0, "end")
                entry.insert(0, selected_path)
                entry.config(fg="black")
                validate_input()  # Validate on file/folder selection

        # width is in pixels on macOS (tkmacosx), in characters on other platforms
        btn_width = 60 if platform.system() == "Darwin" else 8
        browse_button = Button(entry_frame, text="Browse", font=("Helvetica", 9), bg="#f0f0f0",
                               width=btn_width, command=browse_action)
        browse_button.grid(row=0, column=1, padx=(5, 5), pady=2, sticky="e")  # Positioned to the right
        browse_button.bind("<ButtonPress>", lambda e: setattr(entry, '_browsing', True))

    def _make_drop_handler(self, entry, validate_input):
        """Build and return the drag-and-drop handler for an entry field."""
        def handle_drop(event):
            dropped_path = event.data.strip()

            # Remove curly braces if present (from tkinterdnd2 behavior)
            if dropped_path.startswith("{") and dropped_path.endswith("}"):
                dropped_path = dropped_path[1:-1]

            entry.delete(0, "end")
            entry.insert(0, dropped_path)
            validate_input()

        return handle_drop

    def create_input_field(self, parent, label_text, placeholder_text, row, variable=None,
                           validation_func=None, error_message="", browse_type=None):
        """
        Create a labeled input field with placeholder functionality, validation, and inline error message.
        Includes an optional browse button integrated into the entry field.
        """
        # Create a label for the input field
        tk.Label(parent, text=label_text, font=self.label_font, bg="#ffffff").grid(
            row=row, column=0, padx=10, pady=5, sticky="e"
        )

        # Create a frame to contain the entry and the browse button
        entry_frame = tk.Frame(parent, bg="#ffffff", highlightthickness=1, highlightbackground="#cccccc")
        entry_frame.grid(row=row, column=1, padx=10, pady=5, sticky="w")

        # Create the entry widget
        entry = tk.Entry(entry_frame, width=25, font=self.label_font, borderwidth=0)
        entry.grid(row=0, column=0, sticky="w", padx=(5, 0))  # Use grid for internal layout

        # Error message label (initially empty)
        error_label = tk.Label(parent, text="", font=("Helvetica", 9), fg="red", bg="#ffffff")
        error_label.grid(row=row, column=2, padx=(5, 10), sticky="w")  # Positioned to the right of the entry

        # Build validation, placeholder, browse, and drag-and-drop behaviours
        validate_input = self._make_validator(entry, error_label, error_message, validation_func)
        self._make_placeholder_bindings(entry, placeholder_text, validate_input)
        self._make_browse_button(entry_frame, entry, browse_type, validate_input)
        self.enable_drag_and_drop(entry_frame, self._make_drop_handler(entry, validate_input))

        # Bind variable (if provided)
        if variable:
            entry.config(textvariable=variable)

        return entry

    def enable_drag_and_drop(self, widget, callback):
        """
        Enable drag-and-drop functionality for a widget.

        :param widget: The widget to enable drag-and-drop on.
        :param callback: The function to call when a file/folder is dropped.
        """
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", callback)

    def remove_focus(self, event):
        """Remove focus from the currently focused widget unless it's an input widget."""
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
        """Create the options menu with theme toggle functionality."""
        # Create the top menu bar
        menubar = tk.Menu(self)

        # File Menu
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
            ("O-C Plotting", self.show_oc_plot),
            ("Gaia Search", self.show_gaia_query),
            ("O'Connell Effect", self.show_oconnell_effect),
            ("Color Light Curve", self.show_color_light_curve),
            ("Close Program", self.quit_program),
        ]

        for option, command in options:
            self.create_menu_button(option, command)

    def show_settings_menu(self):
        """Show the settings menu for theme toggling."""
        self.clear_right_frame()

        # High Contrast Mode toggle
        high_contrast_var = tk.BooleanVar(value=self.theme_settings["high_contrast"])
        tk.Checkbutton(
            self.right_frame,
            text="High Contrast Mode",
            variable=high_contrast_var,
            command=lambda: self.toggle_theme("high_contrast", high_contrast_var.get()),
            font=("Helvetica", 12),
            bg="#ffffff",
        ).pack(pady=10, anchor="w")

        # Dark Theme toggle
        dark_theme_var = tk.BooleanVar(value=self.theme_settings["dark_theme"])
        tk.Checkbutton(
            self.right_frame,
            text="Dark Theme",
            variable=dark_theme_var,
            command=lambda: self.toggle_theme("dark_theme", dark_theme_var.get()),
            font=("Helvetica", 12),
            bg="#ffffff",
        ).pack(pady=10, anchor="w")

    def toggle_theme(self, theme_type, value):
        """Toggle a theme and update settings."""
        self.theme_settings[theme_type] = value

        # Disable one theme if the other is enabled
        if theme_type == "high_contrast" and value:
            self.theme_settings["dark_theme"] = False
        elif theme_type == "dark_theme" and value:
            self.theme_settings["high_contrast"] = False

        # Apply theme changes
        self.apply_theme()

        # Save settings
        self.save_settings()

    def apply_theme(self):
        """Apply the selected theme to the GUI."""
        high_contrast = self.theme_settings["high_contrast"]
        dark_theme = self.theme_settings["dark_theme"]

        if high_contrast:
            self.configure(bg="black")
            self.left_frame.configure(bg="yellow")
            self.right_frame.configure(bg="yellow")
            font_color = "black"
            button_bg = "yellow"
        elif dark_theme:
            self.configure(bg="#333333")
            self.left_frame.configure(bg="#444444")
            self.right_frame.configure(bg="#555555")
            font_color = "white"
            button_bg = "#666666"
        else:
            # Default theme
            self.configure(bg="#f5f5f5")
            self.left_frame.configure(bg="#f5f5f5")
            self.right_frame.configure(bg="#ffffff")
            font_color = "black"
            button_bg = "#003366"

        # Update all widgets
        for widget in self.left_frame.winfo_children():
            if isinstance(widget, Button):
                widget.configure(bg=button_bg, fg=font_color)

        for widget in self.right_frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.configure(bg=self.right_frame["bg"], fg=font_color)
            elif isinstance(widget, tk.Checkbutton):
                widget.configure(bg=self.right_frame["bg"], fg=font_color, selectcolor=self.right_frame["bg"])

    def save_settings(self):
        """Save current theme settings to a config file."""
        with open(self.config_file, "w") as file:
            json.dump(self.theme_settings, file)

    def load_settings(self):
        """Load theme settings from a config file."""
        try:
            with open(self.config_file, "r") as file:
                self.theme_settings = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.theme_settings = {"high_contrast": False, "dark_theme": False}

    def create_menu_button(self, text, command):
        """Create a menu button"""
        Button(self.left_frame, text=text, command=command,
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
            "Here you can find brief descriptions on each of the currently available options.\n"
            "\nFeatures:\n"
            "- IRAF Reduction: Process raw astronomical images.\n"
            "- TESS Search: Retrieve TESS sector data.\n"
            "- AIJ Comparison: Select comparison stars.\n"
            "- Multi Aperture Photometry: Analysis science images with multi aperture photometry.\n"
            "- Gaia Search: Query Gaia for numerous variables related to an eclipsing binary star.\n"
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
        Button(parent, text="Run", font=self.button_font,
               bg="#003366", fg="white",
               activebackground="#00509e", activeforeground="white",
               relief="flat", cursor="hand2",
               highlightbackground="#003366", highlightthickness=1,
               command=lambda: action(**kwargs)).grid(row=row, column=0, columnspan=2, pady=20)

    def create_cancel_button(self, parent, action, row, **kwargs):
        Button(parent, text="Cancel", font=self.button_font,
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
                                                  r"C:\folder1\raw_images", row=1,
                                                  validation_func=lambda x: len(x.strip()) > 0,
                                                  error_message="File path cannot be empty.",
                                                  browse_type="folder")

        calibrated_images_path = self.create_input_field(self.right_frame, "Calibrated Images Path:",
                                                         r"C:\folder1\calibrated_images", row=2,
                                                         validation_func=lambda x: len(x.strip()) > 0,
                                                         error_message="File path cannot be empty.",
                                                         browse_type="folder")

        location = self.create_input_field(self.right_frame, "Location:",
                                           "e.g., BSUO, CTIO, etc.", row=3,
                                           validation_func=lambda x: isinstance(x, str) and x.isalpha(),
                                           error_message="Value must be a string.")

        # Checkbox for dark frames
        dark_bool_var = tk.BooleanVar(value=True)
        self.create_checkbox(self.right_frame, "Use Dark Frames", dark_bool_var, row=4)

        # Overscan and Trim region inputs
        overscan = self.create_input_field(self.right_frame, "Overscan Region:",
                                           "[2073:2115, :]", row=5,
                                           validation_func=lambda x: len(x.strip()) > 0,
                                           error_message="Please enter at least [:,:].")

        trim = self.create_input_field(self.right_frame, "Trim Region:",
                                       "[20:2060, 12:2057]", row=6,
                                       validation_func=lambda x: len(x.strip()) > 0,
                                       error_message="Please enter at least [:,:].")

        # Button to open and plot bias image
        Button(self.right_frame, text="Open Bias Image", font=self.button_font, bg="#003366", fg="white",
               command=self.open_bias_image).grid(row=7, column=0, columnspan=2, pady=10, sticky="")

        # Run button
        self.create_run_button(self.right_frame, self.run_iraf_reduction, row=8,
                               raw_images_path=raw_images_path,
                               calibrated_images_path=calibrated_images_path,
                               location=location,
                               dark_bool_var=dark_bool_var,
                               overscan_var=overscan,
                               trim_var=trim)

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
                                              validation_func=lambda x: len(x.strip()) > 0,
                                              error_message="Please enter a system name.")

        download_path = self.create_input_field(self.right_frame, "Download Path:",
                                                r"C:\folder1\download", row=2,
                                                validation_func=lambda x: len(x.strip()) > 0,
                                                error_message="Please enter a file pathway.",
                                                browse_type="folder")

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
                self.sector_label = tk.Label(self.right_frame, text="Select Specific Sector:",
                                             font=self.label_font, bg="#ffffff")
                self.sector_label.grid(row=5, column=0, sticky="e")

            # Add sector dropdown
            if not self.sector_dropdown:
                self.sector_dropdown = ttk.Combobox(self.right_frame, state="readonly",
                                                    values=[], font=self.label_font)
                self.sector_dropdown.grid(row=5, column=1, padx=10, pady=5, sticky="w")

            # Add "Retrieve Sectors" button
            if not self.retrieve_button:
                self.retrieve_button = Button(
                    self.right_frame, text="Retrieve Sectors", font=self.button_font,
                    bg="#003366", fg="white",
                    command=lambda: self.retrieve_sectors(
                        system_name=system_name, sector_dropdown=self.sector_dropdown)
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
                                     validation_func=lambda x: len(x.strip()) > 0,
                                     error_message="Please enter a RA.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      validation_func=lambda x: len(x.strip()) > 0,
                                      error_message="Please enter a DEC.")

        folder_path = self.create_input_field(self.right_frame, "Data Save Folder Path:",
                                              r"C:\folder1\download", row=3,
                                              validation_func=lambda x: len(x.strip()) > 0,
                                              error_message="Please enter a file pathway.",
                                              browse_type="folder")

        obj_name = self.create_input_field(self.right_frame, "Object Name:",
                                           "NSVS 896797", row=4,
                                           validation_func=lambda x: len(x.strip()) > 0,
                                           error_message="Please enter the object name.")

        science_image = self.create_input_field(self.right_frame, "Science Image File:",
                                                r"C:\folder1\calibrated_images", row=5,
                                                validation_func=lambda x: len(x.strip()) > 0,
                                                error_message="Please enter a file pathway.",
                                                browse_type="file")

        # Buttons for comparison selector
        Button(self.right_frame, text="Run Comparison Selector", font=self.button_font, bg="#003366", fg="white",
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
                                           validation_func=lambda x: len(x.strip()) > 0,
                                           error_message="Please enter an object name.")

        reduced_images_path = self.create_input_field(self.right_frame, "Reduced Images Path:",
                                                      r"C:\folder1\reduced_images", row=2,
                                                      validation_func=lambda x: len(x.strip()) > 0,
                                                      error_message="Please enter a file pathway.",
                                                      browse_type="folder")

        radec_b_file = self.create_input_field(self.right_frame, "RADEC File (B Filter):",
                                               r"C:\folder1\B.radec", row=3,
                                               validation_func=lambda x: len(x.strip()) > 0,
                                               error_message="Please enter a file pathway with file name",
                                               browse_type="folder")

        radec_v_file = self.create_input_field(self.right_frame, "RADEC File (V Filter):",
                                               r"C:\folder1\V.radec", row=4,
                                               validation_func=lambda x: len(x.strip()) > 0,
                                               error_message="Please enter a file pathway with file name.",
                                               browse_type="folder")

        radec_r_file = self.create_input_field(self.right_frame, "RADEC File (R Filter):",
                                               r"C:\folder1\R.radec", row=5,
                                               validation_func=lambda x: len(x.strip()) > 0,
                                               error_message="Please enter a file pathway with file name.",
                                               browse_type="folder")

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
    
    
    def show_oc_plot(self):
        """Display the O-C Plotting panel."""
        self.clear_right_frame()
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)
    
        tk.Label(self.right_frame, text="O-C Plotting", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )
    
        # Mode selector
        tk.Label(self.right_frame, text="Data Source:", font=self.label_font, bg="#ffffff").grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.oc_mode_var = tk.StringVar(value="BSUO")
        mode_frame = tk.Frame(self.right_frame, bg="#ffffff")
        mode_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        for label, val in [("BSUO/SARA", "BSUO"), ("TESS", "TESS"), ("All Data", "ALL")]:
            tk.Radiobutton(
                mode_frame, text=label, variable=self.oc_mode_var, value=val,
                font=self.label_font, bg="#ffffff",
                command=self._update_oc_fields
            ).pack(side="left", padx=5)
    
        # Has epoch checkbox
        self.oc_has_epoch_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.right_frame, text="I already have an Epoch value",
            variable=self.oc_has_epoch_var, font=self.label_font, bg="#ffffff"
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")
    
        # Shared fields
        self.oc_epoch_entry = self.create_input_field(
            self.right_frame, "Epoch (T0):", "e.g. 2457143.761819", row=3,
            validation_func=lambda x: x.replace(".", "", 1).lstrip("-").isdigit(),
            error_message="Please enter a valid float."
        )
        self.oc_epoch_err_entry = self.create_input_field(
            self.right_frame, "Epoch Error:", "e.g. 0.0002803", row=4,
            validation_func=lambda x: x.replace(".", "", 1).isdigit(),
            error_message="Please enter a valid float."
        )
        self.oc_period_entry = self.create_input_field(
            self.right_frame, "Period (days):", "e.g. 0.31297", row=5,
            validation_func=lambda x: x.replace(".", "", 1).isdigit(),
            error_message="Please enter a valid period."
        )
        self.oc_output_entry = self.create_input_field(
            self.right_frame, "Output Folder:", r"C:\folder\output", row=6,
            validation_func=lambda x: len(x.strip()) > 0,
            error_message="Please enter an output folder path.",
            browse_type="folder"
        )
    
        # BSUO fields
        self.oc_b_file_entry = self.create_input_field(
            self.right_frame, "B Filter ToM File:", r"C:\folder\B_min.txt", row=7,
            validation_func=lambda x: len(x.strip()) > 0,
            error_message="Please enter a file path.",
            browse_type="file"
        )
        self.oc_v_file_entry = self.create_input_field(
            self.right_frame, "V Filter ToM File:", r"C:\folder\V_min.txt", row=8,
            validation_func=lambda x: len(x.strip()) > 0,
            error_message="Please enter a file path.",
            browse_type="file"
        )
        self.oc_r_file_entry = self.create_input_field(
            self.right_frame, "R Filter ToM File:", r"C:\folder\R_min.txt", row=9,
            validation_func=lambda x: len(x.strip()) > 0,
            error_message="Please enter a file path.",
            browse_type="file"
        )
    
        # TESS field
        self.oc_tess_file_entry = self.create_input_field(
            self.right_frame, "TESS ToM File:", r"C:\folder\tess_min.txt", row=7,
            validation_func=lambda x: len(x.strip()) > 0,
            error_message="Please enter a file path.",
            browse_type="file"
        )
    
        # All Data fields
        self.oc_nights_entry = self.create_input_field(
            self.right_frame, "Number of Files:", "e.g. 2", row=7,
            validation_func=lambda x: x.isdigit(),
            error_message="Please enter a whole number."
        )
        self.oc_all_files_entry = self.create_input_field(
            self.right_frame, "O-C Files (comma separated):", r"C:\folder\file1.txt, C:\folder\file2.txt", row=8,
            validation_func=lambda x: len(x.strip()) > 0,
            error_message="Please enter file paths separated by commas."
        )
    
        # Show correct fields for default mode
        self._update_oc_fields()
    
        # Run button
        self.create_run_button(self.right_frame, self.run_oc_plot, row=10)
    
        # Log
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=11, column=0, columnspan=2, pady=5
        )
        self.create_scrollbar_and_log(12)
    

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
                                     validation_func=lambda x: len(x.strip()) > 0,
                                     error_message="Please enter a RA.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      validation_func=lambda x: len(x.strip()) > 0,
                                      error_message="Please enter a DEC.")

        output_file = self.create_input_field(self.right_frame, "Output File Path:",
                                              r"C:\folder1\Gaia_[star name].txt", row=3,
                                              validation_func=lambda x: len(x.strip()) > 0,
                                              error_message="Please enter a file pathway.",
                                              browse_type="folder")

        # Run button
        self.create_run_button(self.right_frame, self.run_gaia_query, row=4,
                               ra=ra, dec=dec, output_file=output_file)

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=5, column=0, columnspan=2, pady=5
        )

        # Create scrollbar for the log area
        self.create_scrollbar_and_log(6)

    def show_oconnell_effect(self):
        """Display the O'Connell Effect panel."""
        self.clear_right_frame()

        # Configure grid for the right frame
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

        # List to store all entry field components (label, entry frame)
        file_entry_widgets = []

        # Function to dynamically display entry fields
        def update_file_path_fields():
            selected_count = filter_count_var.get()  # Get the selected filter count

            # Remove all widgets first
            for widgets in file_entry_widgets:
                for widget in widgets:
                    widget.grid_forget()  # Hide it properly (alternative to destroy)

            file_entry_widgets.clear()  # Reset the list

            # Recreate only the required number of entries
            # filter_list = ["B", "V", "R"]
            for i in range(selected_count):
                # Label for the file path
                label = tk.Label(self.right_frame, text=f"File Path {i + 1}:", font=self.label_font, bg="#ffffff")
                label.grid(row=3 + i, column=0, padx=10, pady=2, sticky="e")

                # Create a frame for entry + browse button (keeps them together)
                entry_frame = tk.Frame(self.right_frame, bg="#ffffff", highlightthickness=1,
                                       highlightbackground="#cccccc")
                entry_frame.grid(row=3 + i, column=1, padx=10, pady=2, sticky="w")

                # Entry widget (set width to match output file path)
                entry = tk.Entry(entry_frame, textvariable=file_path_vars[i], font=self.label_font, width=30,
                                 borderwidth=0)
                entry.grid(row=0, column=0, sticky="w", padx=(5, 0), ipadx=10, ipady=3)

                # Browse button inside the entry frame
                browse_button = Button(
                    entry_frame, text="Browse", font=("Helvetica", 9), bg="#f0f0f0",
                    command=lambda var=file_path_vars[i]: var.set(filedialog.askopenfilename(title="Select File"))
                )
                browse_button.grid(row=0, column=1, padx=(5, 5), pady=2, sticky="e")

                # Store references to all elements
                file_entry_widgets.append((label, entry_frame))

        # Radio buttons for selecting filter count
        tk.Label(self.right_frame, text="Select Number of Filters:", font=self.label_font, bg="#ffffff").grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )

        # Frame for radio buttons (reduces spacing)
        radio_frame = tk.Frame(self.right_frame, bg="#ffffff")
        radio_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        for i in range(1, 4):
            tk.Radiobutton(
                radio_frame,
                text=f"{i} Filter{'s' if i > 1 else ''}",
                variable=filter_count_var,
                value=i,
                command=update_file_path_fields,
                font=self.label_font,
                bg="#ffffff",
                anchor="w"
            ).pack(side="left", padx=5)  # Adjust padding to reduce spacing

        # Initially display the correct number of fields
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
        period_var = self.create_input_field(
            parent=self.right_frame,
            label_text="Period:",
            placeholder_text="0.3175",
            row=7,
            validation_func=lambda x: x.replace(".", "", 1).isdigit(),  # Is a float
            error_message="Please enter a valid Period."
        )

        # System name input
        obj_name_var = self.create_input_field(
            parent=self.right_frame,
            label_text="System Name:",
            placeholder_text="NSVS_896797",
            row=8,
            validation_func=lambda x: len(x.strip()) > 0,  # Ensure not empty
            error_message="Please enter a System Name."
        )

        # Output file path input
        output_var = self.create_input_field(
            parent=self.right_frame,
            label_text="Output File Path:",
            placeholder_text="C:/folder1/folder2",
            row=9,
            validation_func=lambda x: len(x.strip()) > 0,  # Ensure not empty
            error_message="Please enter a file pathway.",
            browse_type="folder"
        )

        # Button to run O'Connell Effect calculation
        Button(
            self.right_frame,
            text="Run O'Connell Effect",
            font=self.button_font,
            bg="#003366",
            fg="white",
            command=lambda: self.run_oconnell_effect(
                filter_count_var.get(),
                [var.get() for var in file_path_vars[:filter_count_var.get()]],  # Get actual file paths
                hjd_var, period_var, obj_name_var, output_var
            )
        ).grid(row=10, column=0, columnspan=2, pady=20)

        # Create scrollbar for the log area
        self.create_scrollbar_and_log(11)

    def show_color_light_curve(self):
        """Display the Color Light Curve panel."""
        self.clear_right_frame()
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        tk.Label(self.right_frame, text="Color Light Curve", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        bfile = self.create_input_field(self.right_frame, "B-band File:",
                                        r"C:\folder\b_band.txt", row=1,
                                        validation_func=lambda x: len(x.strip()) > 0,
                                        error_message="Please select a valid B-band file.",
                                        browse_type="file")

        vfile = self.create_input_field(self.right_frame, "V-band File:",
                                        r"C:\folder\v_band.txt", row=2,
                                        validation_func=lambda x: len(x.strip()) > 0,
                                        error_message="Please select a valid V-band file.",
                                        browse_type="file")

        period = self.create_input_field(self.right_frame, "Period (days):",
                                         "e.g. 1.234", row=3,
                                         validation_func=lambda x: x.replace('.', '', 1).isdigit(),
                                         error_message="Enter a numeric period.")

        hjd = self.create_input_field(self.right_frame, "HJD (Epoch):",
                                      "e.g. 2459000.123", row=4,
                                      validation_func=lambda x: x.replace('.', '', 1).isdigit(),
                                      error_message="Enter a numeric HJD.")

        outname = self.create_input_field(self.right_frame, "Output Image Name:",
                                          "color_curve.png", row=5,
                                          validation_func=lambda x: x.strip().endswith('.png'),
                                          error_message="Output must be a PNG file.")

        self.create_run_button(self.right_frame, self.run_color_light_curve, row=6,
                               bfile=bfile, vfile=vfile, period=period, hjd=hjd, outname=outname)

        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=7, column=0, columnspan=2, pady=5
        )
        self.create_scrollbar_and_log(8)

    def run_iraf_reduction(self, raw_images_path, calibrated_images_path, location,
                           dark_bool_var, overscan_var, trim_var):
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
                self.write_to_log(f"Trim Region: {trim_region}")
                self.write_to_log(f"Overscan Region: {overscan_region}")
                self.write_to_log("Starting IRAF Reduction...\n")

                # Call the IRAF Reduction script
                run_reduction(
                    path=raw_path, calibrated=calibrated_path, location=loc,
                    cancel_event=self.cancel_event, dark_bool=use_dark_frames,
                    overscan_region=overscan_region, trim_region=trim_region,
                    write_callback=self.write_to_log
                )

                if not self.cancel_event.is_set():
                    messagebox.showinfo("Success", "IRAF Reduction completed successfully!")
                else:
                    messagebox.showinfo("Cancelled", "IRAF Reduction was cancelled.")

            except Exception as e:
                self.write_to_log(f"An error occurred during IRAF Reduction: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

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
                    system_name=system_name_value, download_all=download_all,
                    specific_sector=specific_sector_value, download_path=download_path_value,
                    write_callback=self.write_to_log, cancel_event=self.cancel_event
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
                comparison_selector(ra=ra_value, dec=dec_value, pipeline=False,
                                    folder_path=folder_value, obj_name=obj_value,
                                    science_image=science_image_value,
                                    write_callback=self.write_to_log, cancel_event=self.cancel_event)

                if not self.cancel_event.is_set():
                    self.write_to_log("Comparison Selector completed successfully.")
                else:
                    self.write_to_log("Comparison Selector was canceled.")
            except Exception as e:
                self.write_to_log(
                    f"An error occurred during comparison star selection: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(selector_task)

    def run_multi_aperture_photometry(self, obj_name, reduced_images_path,
                                      radec_b_file, radec_v_file, radec_r_file):
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
                    path=reduced_path_value, pipeline=False,
                    radec_list=[radec_b_path, radec_v_path, radec_r_path],
                    obj_name=obj_name_value,
                    write_callback=self.write_to_log, cancel_event=self.cancel_event
                )

                if not self.cancel_event.is_set():
                    self.write_to_log("Multi-Aperture Photometry completed successfully.")
                else:
                    self.write_to_log("Multi-Aperture Photometry was canceled.")
            except Exception as e:
                self.write_to_log(
                    f"An error occurred during Multi-Aperture Photometry: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        # Run the photometry task in a separate thread
        self.run_task(photometry_task)
    
    
    def run_oc_plot(self):
        """Collect inputs and pass them to OC_plot functions in a background thread."""
    
        def oc_plot():
            try:
                from .OC_plot import BSUO, TESS_OC, all_data, data_fit
                import pandas as pd
    
                self.create_cancel_button(self.right_frame, self.cancel_task, row=10)
    
                mode = self.oc_mode_var.get()
                period_str = self.oc_period_entry.get().strip()
                output = self.oc_output_entry.get().strip()
    
                if not period_str:
                    self.write_to_log("Error: Period is required.")
                    return
                period = float(period_str)
    
                if not output:
                    self.write_to_log("Error: Output folder is required.")
                    return
    
                # Determine T0 and error
                if self.oc_has_epoch_var.get() and mode != "ALL":
                    epoch_str = self.oc_epoch_entry.get().strip()
                    epoch_err_str = self.oc_epoch_err_entry.get().strip()
                    if not epoch_str or not epoch_err_str:
                        self.write_to_log("Error: Epoch and Epoch Error are required.")
                        return
                    T0 = float(epoch_str)
                    T0_err = float(epoch_err_str)
                else:
                    T0 = 0
                    T0_err = 0
    
                if mode == "BSUO":
                    b_path = self.oc_b_file_entry.get().strip()
                    v_path = self.oc_v_file_entry.get().strip()
                    r_path = self.oc_r_file_entry.get().strip()
                    if not all([b_path, v_path, r_path]):
                        self.write_to_log("Error: All three filter files are required.")
                        return
                    db = pd.read_csv(b_path, header=None, sep=r"\s+")
                    dv = pd.read_csv(v_path, header=None, sep=r"\s+")
                    dr = pd.read_csv(r_path, header=None, sep=r"\s+")
                    outfile = BSUO(T0, T0_err, period, db, dv, dr, output,
                                   write_callback=self.write_to_log,
                                   cancel_event=self.cancel_event)
    
                elif mode == "TESS":
                    tess_path = self.oc_tess_file_entry.get().strip()
                    if not tess_path:
                        self.write_to_log("Error: TESS ToM file is required.")
                        return
                    df = pd.read_csv(tess_path, header=None, sep=r"\s+")
                    outfile = TESS_OC(T0, T0_err, period, df, output,
                                      write_callback=self.write_to_log,
                                      cancel_event=self.cancel_event)
    
                elif mode == "ALL":
                    nights_str = self.oc_nights_entry.get().strip()
                    if not nights_str.isdigit():
                        self.write_to_log("Error: Number of files must be a whole number.")
                        return
                    all_files_str = self.oc_all_files_entry.get().strip()
                    file_paths = [p.strip() for p in all_files_str.split(",") if p.strip()]
                    if len(file_paths) != int(nights_str):
                        self.write_to_log(f"Error: Expected {nights_str} files, got {len(file_paths)}.")
                        return
                    outfile = all_data(file_paths, period, output,
                                       write_callback=self.write_to_log,
                                       cancel_event=self.cancel_event)
    
                if outfile and not self.cancel_event.is_set():
                    data_fit(outfile, period,
                             write_callback=self.write_to_log,
                             cancel_event=self.cancel_event)
                    self.write_to_log("O-C plotting completed successfully.")
    
            except Exception as e:
                import traceback
                self.write_to_log(f"Error during O-C plotting: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())
    
        self.run_task(oc_plot)
    

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
                    ra_input=ra_value, dec_input=dec_value, output_path=output_path,
                    write_callback=self.write_to_log, cancel_event=self.cancel_event
                )

            except Exception as e:
                self.write_to_log(f"An error occurred during Gaia Query: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(gaia_query)

    def run_color_light_curve(self, bfile, vfile, period, hjd, outname):
        """Run the color light curve plotting task in a separate thread."""

        def task():
            try:
                self.create_cancel_button(self.right_frame, self.cancel_task, row=6)

                b = bfile.get().strip()
                v = vfile.get().strip()
                per = float(period.get().strip())
                epoch = float(hjd.get().strip())
                out = outname.get().strip()

                self.write_to_log("Starting color light curve processing...")
                self.write_to_log(f"B-band file: {b}")
                self.write_to_log(f"V-band file: {v}")
                self.write_to_log(f"Period: {per}")
                self.write_to_log(f"HJD Epoch: {epoch}")
                self.write_to_log(f"Output: {out}")

                color_plot(Bfile=b, Vfile=v, Epoch=epoch, period=per, outName=out, save=True,
                           write_callback=self.write_to_log, cancel_event=self.cancel_event)

                if not self.cancel_event.is_set():
                    messagebox.showinfo("Success", "Color light curve generated successfully!")
                else:
                    messagebox.showinfo("Cancelled", "Color light curve task was cancelled.")

            except Exception as e:
                self.write_to_log(f"Error: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(task)

    def run_oconnell_effect(self, filter_count, file_path_vars, hjd, period, obj_name, output_file):
        """
        Run the O'Connell Effect calculation based on the selected number of filters
        and provided input values.
        """

        def oconnell_task():
            try:
                filter_count_var = filter_count
                # Collect file paths based on the selected filter count
                file_paths = [var.strip() for var in file_path_vars[:filter_count_var]]
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
                    return
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
                oconnell(
                    filepath=output_file_value, filter_files=list(file_paths),
                    obj_name="OConnell_Output", period=float(period_value),
                    hjd=float(hjd_value), write_callback=self.write_to_log,
                    cancel_event=self.cancel_event
                )

                # Notify success
                self.write_to_log("O'Connell Effect calculation completed successfully.")
                messagebox.showinfo("Success", "O'Connell Effect calculation completed successfully!")

            except Exception as e:
                # Log and notify errors
                self.write_to_log(
                    f"An error occurred during O'Connell Effect calculation: {type(e).__name__}: {e}")
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


def main():
    app = ProgramLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
