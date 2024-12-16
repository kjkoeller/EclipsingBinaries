"""
 A splash screen that greets the user when they first start the package and displays a progressbar of the packages
 being imported

Author: Kyle Koeller
Created: 12/16/2024
Last Updated: 12/16/2024
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time


def import_packages(progress_queue):
    """
    Simulate importing packages and update progress.
    Replace `time.sleep()` with actual imports in real scenarios.
    """
    packages = [
        "Importing tkinter...",
        "Importing pathlib...",
        "Importing astropy...",
        "Importing matplotlib...",
        "Importing Custom Scripts...",
        "Finishing..."
    ]
    total = len(packages)

    try:
        # Simulate importing with delays
        import tkinter.messagebox  # Simulate step 1
        progress_queue.put((1, total, packages[0]))
        time.sleep(0.5)

        from pathlib import Path  # Simulate step 2
        progress_queue.put((2, total, packages[1]))
        time.sleep(0.5)

        from astropy.nddata import CCDData  # Simulate step 3
        progress_queue.put((3, total, packages[2]))
        time.sleep(0.5)

        from matplotlib import pyplot as plt  # Simulate step 4
        progress_queue.put((4, total, packages[3]))
        time.sleep(0.5)

        # Custom scripts
        from IRAF_Reduction import run_reduction
        from tess_data_search import run_tess_search
        from apass import comparison_selector
        from multi_aperture_photometry import main as multi_ap
        progress_queue.put((5, total, packages[4]))
        time.sleep(0.5)

    except Exception as e:
        print(f"Error importing packages: {e}")

    # Signal completion
    progress_queue.put(None)


class SplashScreen(tk.Toplevel):
    """
    Splash Screen class to display the progress bar and messages during package loading.
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

        # Splash screen UI elements
        self.title_label = tk.Label(self, text="EclipsingBinaries", font=("Helvetica", 20, "bold"), fg="white", bg="#003366")
        self.title_label.pack(pady=10)

        self.message_label = tk.Label(self, text="Initializing...", font=("Helvetica", 14), fg="white", bg="#003366")
        self.message_label.pack(pady=10)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=int(splash_width * 0.8), mode="determinate")
        self.progress.pack(pady=20)

        # Start monitoring the queue
        self.after(100, self.monitor_progress)

    def monitor_progress(self):
        """
        Monitor the progress queue and update the splash screen.
        """
        try:
            progress_data = self.progress_queue.get_nowait()
            if progress_data is None:
                self.on_close_callback()
                self.destroy()  # Close the splash screen when done
            else:
                current, total, message = progress_data
                self.progress["value"] = (current / total) * 100
                self.message_label.config(text=message)
                self.after(100, self.monitor_progress)  # Check again after 100 ms
        except queue.Empty:
            self.after(100, self.monitor_progress)  # Check again if the queue is empty


def launch_main_gui():
    """
    Launch the main GUI after the splash screen.
    """
    from menu import ProgramLauncher  # Import the main GUI
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

    # Start a thread to simulate importing modules
    threading.Thread(target=import_packages, args=(progress_queue,), daemon=True).start()

    # Run the splash screen's event loop
    root.mainloop()

    # Once the splash screen closes, launch the main GUI
    root.destroy()  # Clean up the splash screen's hidden root window
    launch_main_gui()


if __name__ == "__main__":
    main()
