"""
The main program helps centralize all the other programs into one selection routine that can be run and call all
other programs.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 12/06/2024
"""


import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import threading

# Import the updated IRAF Reduction script
from IRAF_GUI import run_reduction
from tess_data_search import run_tess_search


class ProgramLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        # Get screen dimensions
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # Set window size as 60% of screen dimensions
        self.window_width = int(self.screen_width * 0.6)
        self.window_height = int(self.screen_height * 0.65)

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
            ("AIJ Comparison Star Selector", self.dummy_action),
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

    def show_iraf_reduction(self):
        """Display the IRAF reduction panel"""
        self.clear_right_frame()

        # Configure grid for centering
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add a title
        tk.Label(self.right_frame, text="IRAF Reduction", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Input fields with proper alignment
        raw_images_path = self.create_input_field(self.right_frame, "Raw Images Path:", row=1)
        calibrated_images_path = self.create_input_field(self.right_frame, "Calibrated Images Path:", row=2)
        location = self.create_input_field(self.right_frame, "Location (e.g., BSUO, CTIO, KPNO):", row=3)

        # Checkbox for dark frames
        dark_bool_var = tk.BooleanVar(value=True)
        self.create_checkbox(self.right_frame, "Use Dark Frames", dark_bool_var, row=4)

        # Run button
        self.create_run_button(self.right_frame, self.run_iraf_reduction, row=5,
                               raw_images_path=raw_images_path,
                               calibrated_images_path=calibrated_images_path,
                               location=location,
                               dark_bool_var=dark_bool_var)

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=6, column=0, columnspan=2, pady=5
        )

        # Create a frame to hold the log area and scrollbar
        log_frame = tk.Frame(self.right_frame, bg="#ffffff")
        log_frame.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # Add the Text widget (log area) and the Scrollbar
        self.log_area = tk.Text(log_frame, wrap="word", height=12, font=("Helvetica", 10))
        scrollbar = tk.Scrollbar(log_frame, command=self.log_area.yview)

        # Configure the Text widget to work with the scrollbar
        self.log_area.configure(yscrollcommand=scrollbar.set)

        # Pack the Text widget and the scrollbar
        self.log_area.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure grid weight to allow expansion
        self.right_frame.grid_rowconfigure(7, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

    def show_tess_search(self):
        """Display the TESS Database Search panel."""
        self.clear_right_frame()

        # Configure grid for centering
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        # Add a title
        tk.Label(self.right_frame, text="TESS Database Search", font=self.header_font, bg="#ffffff").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

        # Input fields with proper alignment
        system_name = self.create_input_field(self.right_frame, "System Name (TIC ID):", row=1)
        download_path = self.create_input_field(self.right_frame, "Download Path:", row=2)

        # Radio buttons for download option
        download_all_var = tk.BooleanVar(value=True)
        tk.Radiobutton(
            self.right_frame, text="Download All Sectors", variable=download_all_var, value=True,
            font=self.label_font, bg="#ffffff"
        ).grid(row=3, column=0, columnspan=2, pady=5, sticky="")
        tk.Radiobutton(
            self.right_frame, text="Download Specific Sector", variable=download_all_var, value=False,
            font=self.label_font, bg="#ffffff"
        ).grid(row=4, column=0, columnspan=2, pady=5, sticky="")

        # Specific sector input
        specific_sector = self.create_input_field(self.right_frame, "Specific Sector (if applicable):", row=5)

        # Run button
        self.create_run_button(self.right_frame, self.run_tess_search, row=6,
                               system_name=system_name,
                               download_path=download_path,
                               download_all_var=download_all_var,
                               specific_sector=specific_sector)

        # Log display area
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg="#ffffff").grid(
            row=7, column=0, columnspan=2, pady=5
        )

        self.log_area = tk.Text(self.right_frame, wrap="word", height=12, font=("Helvetica", 10))
        scrollbar = tk.Scrollbar(self.right_frame, command=self.log_area.yview)
        self.log_area.configure(yscrollcommand=scrollbar.set)

        self.log_area.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        scrollbar.grid(row=8, column=2, sticky="ns")

        # Configure scrollable log area
        self.right_frame.grid_rowconfigure(8, weight=1)

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

    def run_iraf_reduction(self, raw_images_path, calibrated_images_path, location, dark_bool_var):
        """Run the IRAF reduction process in a separate thread"""

        def reduction_task():
            try:
                raw_path = Path(raw_images_path.get().strip())
                calibrated_path = Path(calibrated_images_path.get().strip())
                loc = location.get().strip()
                use_dark_frames = dark_bool_var.get()

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
                run_reduction(path=raw_path, calibrated=calibrated_path, location=loc,
                              dark_bool=use_dark_frames, write_callback=self.write_to_log)

                # self.write_to_log("IRAF Reduction completed successfully!")
                messagebox.showinfo("Success", "IRAF Reduction completed successfully!")
            except Exception as e:
                self.write_to_log(f"An error occurred: {e}")
                messagebox.showerror("Error", f"An error occurred: {e}")

        # Run the reduction in a separate thread
        threading.Thread(target=reduction_task, daemon=True).start()

    def run_tess_search(self, system_name, download_path, download_all_var, specific_sector):
        """Run the TESS Database Search and TESSCut processing in a separate thread."""

        def search_task():
            try:
                system_name_value = system_name.get().strip()
                download_path_value = download_path.get().strip()  # Correctly retrieve the string value
                download_all = download_all_var.get()
                specific_sector_value = int(specific_sector.get().strip()) if not download_all else None

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
                if specific_sector_value:
                    self.write_to_log(f"Specific Sector: {specific_sector_value}")

                # Run TESS search
                run_tess_search(
                    system_name=system_name_value,
                    download_all=download_all,
                    specific_sector=specific_sector_value,
                    download_path=download_path_value,  # Pass the corrected path
                    write_callback=self.write_to_log
                )

                self.write_to_log("TESS Database Search completed successfully.")
            except Exception as e:
                self.write_to_log(f"An error occurred during TESS Database Search: {e}")

        # Run the search in a separate thread
        threading.Thread(target=search_task, daemon=True).start()

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
