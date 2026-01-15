import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk
import io
import threading
import time
import math

from src.data_pixel_tensor import NORMALS_NAMES
from src.utils.im_transform import resize_with_padding


DRIVE_NAMES_TO_PATH = {
    "drive1tb": "larger_disk",
    "drive2tb": "disk2tb"
}


class CameraControlGUI:
    """
    A Tkinter-based GUI for controlling a remote FastAPI camera server.

    This application provides a user interface to connect to the server,
    view the live video feed, adjust camera parameters, and manage recording.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Control GUI")
        # Set a minimum size for the window
        self.root.minsize(960, 540)

        # --- State Variables ---
        self.is_streaming = False
        self.is_recording = False
        self.server_url = ""

        # --- Main Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Create two main columns: video and controls
        main_frame.grid_columnconfigure(0, weight=3) # Video panel takes more space
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        self._create_video_panel(main_frame)
        self._create_controls_panel(main_frame)

        # Protocol for closing the window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_video_panel(self, parent):
        """Creates the left panel for video display."""
        video_panel = ttk.Frame(parent, padding="5")
        video_panel.grid(row=0, column=0, sticky="nsew")
        video_panel.grid_rowconfigure(0, weight=1)
        video_panel.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_panel, text="Not Connected",
                                     anchor="center", background="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Element selector for the /capture endpoint
        element_frame = ttk.Frame(video_panel)
        element_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Label(element_frame, text="Display Element:").pack(side=tk.LEFT,
                                                              padx=5)
        self.element_var = tk.StringVar(value="raw")
        self.element_selector = ttk.Combobox(
            element_frame, textvariable=self.element_var, state="readonly",
            values=["raw", 'view_img'] + NORMALS_NAMES
        )
        self.element_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_controls_panel(self, parent):
        """Creates the right panel for all control widgets."""
        controls_panel = ttk.Frame(parent, padding="5")
        controls_panel.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        # --- Connection Frame ---
        conn_frame = ttk.LabelFrame(controls_panel, text="Connection", padding="10")
        conn_frame.pack(fill=tk.X, pady=5)
        self._populate_connection_frame(conn_frame)

        # --- Camera Parameters Frame ---
        params_frame = ttk.LabelFrame(controls_panel, text="Camera Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        self._populate_params_frame(params_frame)

        # --- Recording Frame ---
        rec_frame = ttk.LabelFrame(controls_panel, text="Recording", padding="10")
        rec_frame.pack(fill=tk.X, pady=5)
        self._populate_recording_frame(rec_frame)

    def _populate_connection_frame(self, frame):
        """Populates the connection frame with widgets."""
        frame.grid_columnconfigure(1, weight=1)
        ttk.Label(frame, text="Server IP:").grid(row=0, column=0, sticky="w", pady=2)
        self.ip_var = tk.StringVar(value="localhost")
        ttk.Entry(frame, textvariable=self.ip_var).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Port:").grid(row=1, column=0, sticky="w", pady=2)
        self.port_var = tk.StringVar(value="8000")
        ttk.Entry(frame, textvariable=self.port_var).grid(row=1, column=1, sticky="ew")

        self.connect_button = ttk.Button(frame, text="Connect", command=self.connect_to_server)
        self.connect_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

    def _populate_params_frame(self, frame):
        """Populates the camera parameters frame with widgets."""
        # Configure grid to have a weighted column for the scale
        frame.grid_columnconfigure(1, weight=1)

        # --- Variables ---
        self.exposure_var = tk.DoubleVar()
        self.gain_var = tk.DoubleVar()
        self.pixel_format_var = tk.StringVar()
        self.exposure_mode_var = tk.StringVar()
        self.custom_ae_var = tk.StringVar()

        # --- Labels ---
        ttk.Label(frame, text="Exposure:").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text="Gain:").grid(row=1, column=0, sticky="w")
        ttk.Label(frame, text="Pixel Format:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(frame, text="Exposure Mode:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(frame, text="Custom AE:").grid(row=4, column=0, sticky="w", pady=2)

        # --- Controls ---
        # Add a command to the sliders to update the value labels
        self.exposure_scale = ttk.Scale(frame, orient="horizontal", variable=self.exposure_var, state="disabled", command=self._update_exposure_label)
        self.gain_scale = ttk.Scale(frame, orient="horizontal", variable=self.gain_var, state="disabled", command=self._update_gain_label)
        self.pixel_format_combo = ttk.Combobox(frame, textvariable=self.pixel_format_var, state="disabled")
        self.exposure_mode_combo = ttk.Combobox(frame, textvariable=self.exposure_mode_var, state="disabled")
        self.custom_ae_combo = ttk.Combobox(frame, textvariable=self.custom_ae_var, state="disabled")

        # --- Value Labels ---
        # Labels to display the actual slider values
        self.exposure_value_label = ttk.Label(frame, text="-", width=8, anchor="w")
        self.gain_value_label = ttk.Label(frame, text="-", width=8, anchor="w")

        # --- Grid Layout ---
        self.exposure_scale.grid(row=0, column=1, sticky="ew")
        self.exposure_value_label.grid(row=0, column=2, padx=(5, 0))

        self.gain_scale.grid(row=1, column=1, sticky="ew")
        self.gain_value_label.grid(row=1, column=2, padx=(5, 0))

        # The comboboxes span two columns to align with scale + value label
        self.pixel_format_combo.grid(row=2, column=1, columnspan=2, sticky="ew")
        self.exposure_mode_combo.grid(row=3, column=1, columnspan=2, sticky="ew")
        self.custom_ae_combo.grid(row=4, column=1, columnspan=2, sticky="ew")

        self.apply_button = ttk.Button(frame, text="Apply Settings", command=self.apply_camera_settings, state="disabled")
        self.apply_button.grid(row=5, column=0, columnspan=3, pady=10, sticky="ew")

    def _update_exposure_label(self, value):
        """Updates the exposure label with the linear value."""
        try:
            # Convert from log scale back to linear for display
            actual_exposure = math.exp(float(value))
            # Display as an integer (microseconds)
            self.exposure_value_label.config(text=f"{actual_exposure:.0f}")
        except (ValueError, OverflowError):
            self.exposure_value_label.config(text="-")

    def _update_gain_label(self, value):
        """Updates the gain label with its direct value."""
        try:
            self.gain_value_label.config(text=f"{float(value):.2f}")
        except ValueError:
            self.gain_value_label.config(text="-")

    def _populate_recording_frame(self, frame):
        """Populates the recording control frame with widgets."""
        frame.grid_columnconfigure(1, weight=1)

        # --- Drive Selection Dropdown (New) ---
        ttk.Label(frame, text="Save Drive:").grid(row=0, column=0, sticky="w", pady=2)
        self.drive_var = tk.StringVar(value="drive1tb")
        self.drive_combo = ttk.Combobox(
            frame,
            textvariable=self.drive_var,
            values=["drive1tb", "drive2tb"],
            state="disabled"  # Initially disabled
        )
        self.drive_combo.grid(row=0, column=1, sticky="ew")

        # --- Focal Length Dropdown ---
        ttk.Label(frame, text="Focal Length:").grid(row=1, column=0, sticky="w", pady=2)
        self.focal_length_var = tk.StringVar(value="8mm")
        self.focal_length_combo = ttk.Combobox(
            frame,
            textvariable=self.focal_length_var,
            values=["8mm", "12mm", "16mm"],
            state="disabled" # Initially disabled
        )
        self.focal_length_combo.grid(row=1, column=1, sticky="ew")

        # --- Record Name Entry ---
        ttk.Label(frame, text="Record Name:").grid(row=2, column=0, sticky="w", pady=2)
        self.record_name_var = tk.StringVar(value=f"rec_{int(time.time())}")
        self.record_name_entry = ttk.Entry(frame, textvariable=self.record_name_var, state="disabled")
        self.record_name_entry.grid(row=2, column=1, sticky="ew")

        # --- Recording Buttons ---
        self.start_rec_button = ttk.Button(frame, text="Start Recording", command=self.start_recording, state="disabled")
        self.start_rec_button.grid(row=3, column=0, pady=5, sticky="ew", padx=(0,2))

        self.stop_rec_button = ttk.Button(frame, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.stop_rec_button.grid(row=3, column=1, pady=5, sticky="ew", padx=(2,0))

        # --- Saved Count Label ---
        self.saved_count_var = tk.StringVar(value="Saved Frames: 0")
        ttk.Label(frame, textvariable=self.saved_count_var).grid(row=4, column=0, columnspan=2)

    def connect_to_server(self):
        """Establishes a connection to the server and populates controls."""
        ip = self.ip_var.get()
        port = self.port_var.get()
        if not ip or not port:
            messagebox.showerror("Error", "IP and Port cannot be empty.")
            return

        self.server_url = f"http://{ip}:{port}"

        try:
            # Ping the server by getting camera info
            response = requests.get(f"{self.server_url}/get_camera_info", timeout=5)
            response.raise_for_status() # Raise an exception for bad status codes

            camera_info = response.json()
            self._configure_controls(camera_info)

            messagebox.showinfo("Success", "Connected to camera server.")
            self.connect_button.config(text="Disconnect", command=self.disconnect_from_server)
            self._set_controls_state("normal")

            self.apply_camera_settings()

            # Start video stream
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self.video_stream_loop, daemon=True)
            self.stream_thread.start()

        except requests.exceptions.RequestException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to server: {e}")
            self.server_url = ""

    def _configure_controls(self, camera_info):
        """Uses info from the server to configure the control widgets."""
        # --- Exposure (Logarithmic Scale) ---
        exp_min = camera_info.get('exposure_min', 1.0)
        exp_max = camera_info.get('exposure_max', 100000.0)
        # Ensure exp_min is not zero or negative for log
        log_exp_min = math.log(exp_min) if exp_min > 0 else 0
        log_exp_max = math.log(exp_max) if exp_max > 0 else 0
        self.exposure_scale.config(from_=log_exp_min, to=log_exp_max)
        self.exposure_var.set(log_exp_min) # Set slider to minimum

        # --- Gain (Linear Scale) ---
        gain_min = camera_info.get('gain_min', 0.0)
        gain_max = camera_info.get('gain_max', 10.0)
        self.gain_scale.config(from_=gain_min, to=gain_max)
        self.gain_var.set(gain_min) # Set slider to minimum

        # --- Comboboxes ---
        self.pixel_format_combo['values'] = camera_info['available_pixel_format']
        self.exposure_mode_combo['values'] = camera_info['available_exposure_modes']
        self.custom_ae_combo['values'] = camera_info['available_custom_autoexposure_options']
        # Set default selection
        if camera_info['available_pixel_format']: self.pixel_format_var.set(camera_info['available_pixel_format'][0])
        if camera_info['available_exposure_modes']: self.exposure_mode_var.set(camera_info['available_exposure_modes'][0])
        if camera_info['available_custom_autoexposure_options']: self.custom_ae_var.set(camera_info['available_custom_autoexposure_options'][0])

        # Manually trigger an update to set the initial label text
        self._update_exposure_label(self.exposure_var.get())
        self._update_gain_label(self.gain_var.get())

    def _set_controls_state(self, state):
        """Enable or disable all interactive control widgets."""
        widget_state = "normal" if state == "normal" else "disabled"
        combo_state = "readonly" if state == "normal" else "disabled"

        self.exposure_scale.config(state=widget_state)
        self.gain_scale.config(state=widget_state)
        self.pixel_format_combo.config(state=combo_state)
        self.exposure_mode_combo.config(state=combo_state)
        self.custom_ae_combo.config(state=combo_state)
        self.apply_button.config(state=widget_state)
        self.record_name_entry.config(state=widget_state)
        self.start_rec_button.config(state=widget_state)
        self.focal_length_combo.config(state=combo_state)
        self.drive_combo.config(state=combo_state) # Manage drive combo state

        # Also reset value labels on disable
        if state == "disabled":
            self.exposure_value_label.config(text="-")
            self.gain_value_label.config(text="-")

    def disconnect_from_server(self):
        """Disconnects from the server and resets the GUI."""
        self.is_streaming = False
        if self.is_recording:
            self.stop_recording()

        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=1.5)

        self.video_label.config(image='', text="Not Connected")
        self.video_label.image = None

        self._set_controls_state("disabled")
        self.stop_rec_button.config(state="disabled")

        self.connect_button.config(text="Connect", command=self.connect_to_server)
        self.server_url = ""
        messagebox.showinfo("Disconnected", "Disconnected from the server.")

    def video_stream_loop(self):
        """Continuously fetches and displays frames from the server."""
        while self.is_streaming:
            try:
                element = self.element_var.get()
                url = f"{self.server_url}/capture?element={element}"
                response = requests.get(url, stream=True, timeout=5)
                response.raise_for_status()

                pil_image = Image.open(io.BytesIO(response.content))

                max_w, max_h = self.video_label.winfo_width(), self.video_label.winfo_height()
                # Prevent trying to resize to 0x0 on startup
                if max_w < 1 or max_h < 1:
                    time.sleep(0.1)
                    continue

                new_size = (int(max_w * 0.95), int(max_h * 0.95))

                resized_image = resize_with_padding(pil_image, new_size, fill_color=0)
                tk_image = ImageTk.PhotoImage(resized_image)

                # Update label in the main thread
                self.video_label.config(image=tk_image)
                self.video_label.image = tk_image  # Keep a reference!

            except requests.exceptions.RequestException as e:
                print(f"Stream error: {e}")
                time.sleep(1)  # Wait a bit before retrying
            except Exception as e:
                print(f"An unexpected error occurred in video stream: {e}")
                time.sleep(0.1)

    def apply_camera_settings(self):
        """Sends selected camera parameters to the server."""
        # Convert the logarithmic slider value back to a linear value for the server
        log_exposure = self.exposure_var.get()
        actual_exposure = math.exp(log_exposure)

        params = {
            "exposure": actual_exposure,
            "gain": self.gain_var.get(),
            "pixel_format": self.pixel_format_var.get(),
            "exposure_mode": self.exposure_mode_var.get(),
            "custom_exposure_mode": self.custom_ae_var.get()
        }
        # Filter out empty or default values that the user hasn't touched
        payload = {k: v for k, v in params.items() if v}

        if not payload:
            messagebox.showinfo("Info", "No new parameters selected to apply.")
            return

        try:
            response = requests.post(f"{self.server_url}/set_camera_params", json=payload, timeout=5)
            response.raise_for_status()
            messagebox.showinfo("Success", "Camera parameters updated.")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to set parameters: {e}")

    def start_recording(self):
        """Sends a request to start recording frames on the server."""
        user_record_name = self.record_name_var.get()
        if not user_record_name:
            messagebox.showerror("Error", "Record Name cannot be empty.")
            return

        # Prepend drive and focal length to the record name
        drive = self.drive_var.get()
        focal_length = self.focal_length_var.get()
        # Construct the final record name with the drive as a parent folder
        record_name = f"{DRIVE_NAMES_TO_PATH[drive]}/{focal_length}_{user_record_name}"

        try:
            response = requests.post(f"{self.server_url}/start_recording", json={"record_name": record_name}, timeout=5)
            response.raise_for_status()

            self.is_recording = True
            self.start_rec_button.config(state="disabled")
            self.stop_rec_button.config(state="normal")
            # Freeze the record name, focal length, and drive during recording
            self.record_name_entry.config(state="disabled")
            self.focal_length_combo.config(state="disabled")
            self.drive_combo.config(state="disabled")

            self.update_saved_count() # Start polling for saved frame count
            messagebox.showinfo("Recording", f"Recording started as '{record_name}'.")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")

    def stop_recording(self):
        """Sends a request to stop the current recording."""
        try:
            response = requests.post(f"{self.server_url}/stop_recording",
                                     timeout=10) # Longer timeout
            response.raise_for_status()

            self.is_recording = False  # This stops the polling loop
            self.start_rec_button.config(state="normal")
            self.stop_rec_button.config(state="disabled")
            # Unfreeze the record name, focal length, and drive
            self.record_name_entry.config(state="normal")
            self.focal_length_combo.config(state="readonly")
            self.drive_combo.config(state="readonly")

            self.record_name_var.set(f"rec_{int(time.time())}")  # Suggest a new name
            messagebox.showinfo("Recording", "Recording stopped.")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to stop recording: {e}")
        finally:
            self.is_recording = False  # Ensure this is set even if the request fails

    def update_saved_count(self):
        """Periodically polls the server for the number of saved frames."""
        if not self.is_recording:
            self.saved_count_var.set("Saved Frames: 0")
            return

        try:
            response = requests.get(f"{self.server_url}/get_saved_count", timeout=1)
            if response.ok:
                count = response.json().get("n_records", 0)
                self.saved_count_var.set(f"Saved Frames: {count}")
        except requests.exceptions.RequestException as e:
            print(f"Could not get saved count: {e}")

        # Schedule the next update
        if self.is_recording:
            self.root.after(1000, self.update_saved_count)  # Poll every second

    def on_closing(self):
        """Handles the window close event for a graceful shutdown."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.is_streaming = False
            self.is_recording = False  # To stop the count updater
            if self.server_url and self.is_recording:
                try:
                    # Non-blocking attempt to stop recording
                    requests.post(f"{self.server_url}/stop_recording", timeout=2)
                except requests.exceptions.RequestException:
                    pass # Ignore errors on close
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraControlGUI(root)
    root.mainloop()
