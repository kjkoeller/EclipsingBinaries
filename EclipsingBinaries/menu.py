"""
The main program helps centralize all the other programs into one selection routine that can be run and call all
other programs.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 12/11a/2024
"""


import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from pathlib import Path
import threading
from astropy.nddata import CCDData
from matplotlib import pyplot as plt
# from astropy.io import fits
import pandas as pd
import traceback

# Import the updated IRAF Reduction script

from .IRAF_Reduction import run_reduction
from .tess_data_search import run_tess_search
from .apass import comparison_selector


class ProgramLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        # Get screen dimensions
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # Set window size as 60% of screen dimensions
        self.window_width = int(self.screen_width * 0.6)
        self.window_height = int(self.screen_height * 0.7)

        # Center the window
        self.center_window()

        # Window setup
        self.title("EclipsingBinaries")
        self.configure(bg="#f5f5f5")

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
        """Create the options menu"""
        options = [
            ("IRAF Reduction", self.show_iraf_reduction),
            ("Find Minimum (WIP)", self.dummy_action),
            ("TESS Database Search/Download", self.show_tess_search),
            ("AIJ Comparison Star Selector", self.show_aij_comparison_selector),
            ("Multi-Aperture Calculation", self.dummy_action),
            ("BSUO or SARA/TESS Night Filters", self.dummy_action),
            ("O-C Plotting", self.dummy_action),
            ("Gaia Search", self.dummy_action),
            ("O'Connell Effect", self.dummy_action),
            ("Color Light Curve", self.dummy_action),
            ("Close Program", self.quit_program)
        ]

        for option, command in options:
            self.create_menu_button(option, command)

    def create_menu_button(self, text, command):
        """Create a menu button"""
        tk.Button(self.left_frame, text=text, command=command,
                  font=self.button_font, bg="#003366", fg="white", activebackground="#00509e",
                  activeforeground="white", relief="flat", cursor="hand2").pack(pady=5, padx=10, fill="x")

    def cancel_task(self):
        """Cancel the currently running task."""
        if self.current_task and self.current_task.is_alive():
            if messagebox.askyesno("Cancel Task", "Are you sure you want to cancel the current task?"):
                self.write_to_log("Task cancellation requested...")
                self.cancel_event.set()
        else:
            messagebox.showinfo("No Task Running", "There is no task currently running.")

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

        # Input fields
        raw_images_path = self.create_input_field(self.right_frame, "Raw Images Path:", row=1)
        calibrated_images_path = self.create_input_field(self.right_frame, "Calibrated Images Path:", row=2)
        location = self.create_input_field(self.right_frame, "Location (e.g., BSUO, CTIO, etc):", row=3)

        # Checkbox for dark frames
        dark_bool_var = tk.BooleanVar(value=True)
        self.create_checkbox(self.right_frame, "Use Dark Frames", dark_bool_var, row=4)

        # Overscan and Trim region inputs
        overscan = self.create_input_field(self.right_frame, "Overscan Region:", row=5)
        trim = self.create_input_field(self.right_frame, "Trim Region:", row=6)

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

        # create the scroll bar and log area
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
        system_name = self.create_input_field(self.right_frame, "System Name (TIC ID):", row=1)
        download_path = self.create_input_field(self.right_frame, "Download Path:", row=2)

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
            from astroquery.mast import Tesscut
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
                self.sector_label.grid(row=4, column=0, sticky="e")

            # Add sector dropdown
            if not self.sector_dropdown:
                self.sector_dropdown = ttk.Combobox(self.right_frame, state="readonly", values=[], font=self.label_font)
                self.sector_dropdown.grid(row=4, column=1, padx=10, pady=5, sticky="w")

            # Add "Retrieve Sectors" button
            if not self.retrieve_button:
                self.retrieve_button = tk.Button(
                    self.right_frame, text="Retrieve Sectors", font=self.button_font, bg="#003366", fg="white",
                    command=lambda: self.retrieve_sectors(system_name=system_name, sector_dropdown=self.sector_dropdown)
                )
                self.retrieve_button.grid(row=5, column=0, columnspan=2, pady=10)
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
        ra = self.create_input_field(self.right_frame, "Right Ascension (RA):", row=1)
        dec = self.create_input_field(self.right_frame, "Declination (DEC):", row=2)
        folder_path = self.create_input_field(self.right_frame, "Data Save Folder Path:", row=3)
        obj_name = self.create_input_field(self.right_frame, "Object Name:", row=4)
        science_image = self.create_input_field(self.right_frame, "Science Image Folder Path:", row=5)

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

    def write_to_log(self, message):
        """Write a message to the log area and ensure it updates"""
        self.log_area.insert("end", message + "\n")
        self.log_area.see("end")  # Scroll to the latest message
        self.update()  # Process pending GUI events to refresh the log

    def clear_right_frame(self):
        """Clear the right frame by destroying all its widgets"""
        for widget in self.right_frame.winfo_children():
            widget.destroy()

    def create_input_field(self, parent, label_text, row):
        """Create a labeled input field with alignment"""
        tk.Label(parent, text=label_text, font=self.label_font, bg="#ffffff").grid(
            row=row, column=0, padx=10, pady=5, sticky="e"
        )
        entry = tk.Entry(parent, width=40, font=self.label_font)
        entry.grid(row=row, column=1, padx=10, pady=5, sticky="w")
        return entry

    def create_checkbox(self, parent, text, variable, row):
        """Create a checkbox with alignment"""
        tk.Checkbutton(parent, text=text, variable=variable, font=self.label_font, bg="#ffffff").grid(
            row=row, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
        )

    def create_run_button(self, parent, action, row, **kwargs):
        """Create the 'Run' button dynamically with proper alignment"""
        tk.Button(parent, text="Run", font=self.button_font, bg="#003366", fg="white",
                  activebackground="#00509e", activeforeground="white", relief="flat", cursor="hand2",
                  command=lambda: action(**kwargs)).grid(row=row, column=0, columnspan=2, pady=20)

    def create_cancel_button(self, parent, action, row, **kwargs):
        tk.Button(parent, text="Cancel", font=self.button_font, bg="#003366", fg="white",
                  activebackground="#00509e", activeforeground="white", relief="flat", cursor="hand2",
                  command=lambda: action(**kwargs)).grid(row=row, column=1, columnspan=2, pady=20)

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
                self.write_to_log(f"An error occurred: {e}")
                messagebox.showerror("Error", f"An error occurred during IRAF Reduction: {e}")

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
                self.write_to_log(f"An error occurred: {e}")
                messagebox.showinfo("Error", f"An error occurred during TESS Database Search: {e}")

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
                self.write_to_log(f"An error occurred: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(selector_task)

    def dummy_action(self):
        """Dummy action for unimplemented features"""
        messagebox.showinfo("Action", "This feature is not implemented yet.")

    def quit_program(self):
        """Quit the program with confirmation"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.destroy()


if __name__ == "__main__":
    app = ProgramLauncher()
    app.mainloop()