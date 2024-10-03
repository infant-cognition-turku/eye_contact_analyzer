import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import subprocess
import sys
from pathlib import Path


class EyeContactAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Eye Contact Analyzer")
        self.geometry("600x400")

        if self.check_dependencies():
            self.create_variables()
            self.create_widgets()
        else:
            self.create_dependency_message()

    def check_dependencies(self):
        required_packages = {
            'opencv-python': 'cv2',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'torch': 'torch',
            'torchvision': 'torchvision',
            'Pillow': 'PIL'
        }

        missing_packages = []
        for package, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.missing_packages = missing_packages
            return False
        return True

    def create_dependency_message(self):
        message_frame = ttk.Frame(self, padding=20)
        message_frame.pack(expand=True, fill='both')

        ttk.Label(message_frame, text="Missing Required Dependencies",
                  font=('', 12, 'bold')).pack(pady=10)

        message = "The following Python packages are required but not installed:\n\n"
        message += "\n".join(f"• {pkg}" for pkg in self.missing_packages)
        message += "\n\nPlease install these packages and restart the application."

        ttk.Label(message_frame, text=message, wraplength=400).pack(pady=10)
        ttk.Button(message_frame, text="Exit", command=self.quit).pack(pady=10)

    def create_variables(self):
        self.gz_file_path = tk.StringVar()
        self.video_file_path = tk.StringVar()
        self.output_csv = tk.BooleanVar(value=True)
        self.output_video = tk.BooleanVar(value=True)

    def create_widgets(self):
        # File Selection Frame
        file_frame = ttk.LabelFrame(self, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        # Gaze Data File (.gz)
        ttk.Label(file_frame, text="Gaze Data File (.gz):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.gz_file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file('gz')).grid(row=0, column=2)

        # Video File
        ttk.Label(file_frame, text="Video File:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.video_file_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file('video')).grid(row=1, column=2)

        # Output Options Frame
        output_frame = ttk.LabelFrame(self, text="Output Options", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Checkbutton(output_frame, text="Generate Final CSV", variable=self.output_csv).pack(anchor=tk.W)
        ttk.Checkbutton(output_frame, text="Generate Final Video", variable=self.output_video).pack(anchor=tk.W)

        # Process Button
        ttk.Button(self, text="Process", command=self.start_processing).pack(pady=10)

        # Progress Frame
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress = ttk.Progressbar(progress_frame, length=400, mode='indeterminate')
        self.progress.pack()

        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack()

    def browse_file(self, file_type):
        if file_type == 'gz':
            filetypes = [("Gaze Data files", "*.gz"), ("All files", "*.*")]
            var = self.gz_file_path
        else:  # video
            filetypes = [("Video files", "*.mp4"), ("All files", "*.*")]
            var = self.video_file_path

        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            var.set(filename)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.update_idletasks()

    def start_processing(self):
        if not self.gz_file_path.get() or not self.video_file_path.get():
            messagebox.showerror("Error", "Please select both gaze data and video files")
            return

        self.progress.start()
        threading.Thread(target=self.process_files, daemon=True).start()

    def process_files(self):
        try:
            working_dir = os.path.dirname(self.video_file_path.get())

            # Step 1: Process gaze data
            self.update_status("Processing gaze data...")
            subprocess.run([
                sys.executable, "gaze_processor.py",
                "--input", self.gz_file_path.get()
            ], check=True)

            gaze_csv = os.path.join(os.path.dirname(self.gz_file_path.get()), 'gazedata_frames.csv')

            # Step 2: Run face detection
            self.update_status("Running face detection...")
            subprocess.run([
                sys.executable, "face_detector.py",
                "--video", self.video_file_path.get(),
                "--save_csv"
            ], check=True)

            video_name = Path(self.video_file_path.get()).stem
            face_csv = f"{video_name}_output.csv"

            # Create output directory
            output_dir = os.path.join(working_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)

            # Step 3: Final processing
            self.update_status("Generating final outputs...")
            cmd = [
                sys.executable, "final_processor.py",
                "--gaze_csv", gaze_csv,
                "--face_csv", face_csv,
                "--input_video", self.video_file_path.get(),
                "--output_dir", output_dir,
            ]

            if self.output_csv.get():
                cmd.extend(["--generate_csv", "true"])
            if self.output_video.get():
                cmd.extend(["--generate_video", "true"])

            subprocess.run(cmd, check=True)

            self.progress.stop()

            # Prepare success message with output locations
            message = "Processing complete!\n\n"
            if self.output_csv.get():
                message += f"Final CSV file: {os.path.join(output_dir, 'final_output.csv')}\n"
            if self.output_video.get():
                message += f"Final video: {os.path.join(output_dir, 'final_output_video.mp4')}\n"

            messagebox.showinfo("Success", message)
            self.update_status("Ready")

        except subprocess.CalledProcessError as e:
            self.progress.stop()
            messagebox.showerror("Error", f"An error occurred during processing:\n{str(e)}")
            self.update_status("Error occurred")
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
            self.update_status("Error occurred")


if __name__ == "__main__":
    app = EyeContactAnalyzer()
    app.mainloop()