import tkinter as tk
from tkinter import ttk
import threading
import pyvirtualcam
from src.camera import LiveCamera
import platform
import numpy as np
from PIL import Image, ImageTk, ImageOps
from src.data_pixel_tensor import ELEMENTS_NAMES
import math

# Enable high DPI awareness on Windows
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e:
        print("Could not set DPI awareness:", e)


class VirtualCameraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Virtual Camera Controller")
        self.geometry("500x450")
        
        try:
            self.tk.call('tk', 'scaling', 1.5)
        except Exception as e:
            print("Failed to set DPI scaling:", e)
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.camera = LiveCamera(lazy_calculations=True)
        
        # Variable for the selected view from the dropdown.
        self.view_option = tk.StringVar(value="raw")
        self.view_option.trace_add("write", self.update_selected_view)
        self.selected_view = self.view_option.get()
        
        # New variable for camera mode selection.
        self.mode_option = tk.StringVar(value=self.camera.current_exposure_mode)  # default mode
        self.mode_option.trace_add("write", self.update_camera_mode)

        self.preview_enabled = tk.BooleanVar(value=False)
        self.current_frame = None
        self.preview_width = 700
        self.preview_height = 650
        
        self.create_widgets()
        
        self.running = False
        self.camera_thread = None

    def update_selected_view(self, *args):
        self.selected_view = self.view_option.get()

    def update_camera_mode(self, *args):
        mode = self.mode_option.get()
        if self.camera is not None:
            self.camera.set_exposure_mode(mode)
            print(f"Exposure mode set to {mode}")

    def create_widgets(self):
        # Top: View Selection Dropdown.
        view_frame = ttk.LabelFrame(self, text="Select View")
        view_frame.pack(padx=10, pady=10, fill="x")
        
        view_label = ttk.Label(view_frame, text="Select view:")
        view_label.pack(side="left", padx=5, pady=5)
        
        view_dropdown = ttk.Combobox(view_frame,
                                     textvariable=self.view_option,
                                     values=ELEMENTS_NAMES,
                                     state="readonly")
        view_dropdown.pack(side="left", padx=5, pady=5)
        view_dropdown.set("raw")  # default selection

        # Camera Settings Frame (exposure, gain, and mode).
        settings_frame = ttk.LabelFrame(self, text="Camera Settings")
        settings_frame.pack(padx=10, pady=10, fill="x")
        
        # Exposure slider on an exponential (logarithmic) scale.
        exposure_label = ttk.Label(settings_frame, text="Exposure (us) [20 - 1000000]:")
        exposure_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        # Use tk.Scale (instead of ttk.Scale) for finer control.
        # Slider value represents log10(exposure); range: log10(20) ~1.30 to log10(1000000)=6.00.
        self.exposure_slider = ttk.Scale(settings_frame,
                                         from_=math.log10(20),
                                         to=math.log10(1000000),
                                         orient=tk.HORIZONTAL,
                                         length=400,
                                         command=self.update_exposure)
        # Set default slider value to yield ~50,000 us.
        self.exposure_slider.set(math.log10(50000))
        self.exposure_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.exposure_value_label = ttk.Label(settings_frame,
                                              text=f"{int(10 ** float(self.exposure_slider.get()))} us",
                                              width=12, anchor="e")
        self.exposure_value_label.grid(row=0, column=2, sticky="w", padx=5, pady=2)
        
        # Gain slider and value label.
        gain_label = ttk.Label(settings_frame, text="Gain (dB) [0 - 24]:")
        gain_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.gain_slider = ttk.Scale(settings_frame, from_=0, to=24,
                                     orient="horizontal")
        self.gain_slider.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.gain_slider.set(10)  # default value
        self.gain_value_label = ttk.Label(settings_frame,
                                          text=f"{int(self.gain_slider.get())} dB",
                                          width=12, anchor="e")
        self.gain_value_label.grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.gain_slider.config(command=self.update_gain)
        
        # Exposure mode dropdown.
        mode_label = ttk.Label(settings_frame, text="Exposure Mode:")
        mode_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        mode_options = list(self.camera.available_exposure_modes)
        mode_dropdown = ttk.Combobox(settings_frame, textvariable=self.mode_option,
                                     values=mode_options, state="readonly")
        mode_dropdown.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        mode_dropdown.set("off")
        
        # Reset button.
        reset_button = ttk.Button(settings_frame, text="Reset Camera Settings", command=self.reset_camera_settings)
        reset_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        settings_frame.columnconfigure(1, weight=1)
        
        # Bottom container.
        self.bottom_container = ttk.Frame(self)
        self.bottom_container.pack(side="bottom", fill="x", padx=10, pady=10)
        self.bottom_container.columnconfigure(0, weight=1)
        self.bottom_container.columnconfigure(1, weight=0)
        self.bottom_container.columnconfigure(2, weight=1)
        
        # Control Frame: Start/Stop buttons and Preview checkbox.
        self.control_frame = ttk.Frame(self.bottom_container)
        self.control_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        self.control_frame.columnconfigure(0, weight=1)
        
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_camera, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        preview_cb = ttk.Checkbutton(self.control_frame, text="Enable Preview",
                                     variable=self.preview_enabled, command=self.toggle_preview)
        preview_cb.grid(row=0, column=2, padx=5, pady=5)
        
        # Preview Frame.
        self.preview_frame = ttk.Frame(self.bottom_container, width=self.preview_width,
                                       height=self.preview_height, relief="sunken")
        self.preview_frame.grid_propagate(False)
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(fill="both", expand=True)

    def reset_camera_settings(self):
        """Reset the exposure and gain sliders to their default values."""
        default_exposure_val = 50000
        default_gain = 10
        self.exposure_slider.set(math.log10(default_exposure_val))
        self.exposure_value_label.config(text=f"{default_exposure_val} us")
        self.gain_slider.set(default_gain)
        self.gain_value_label.config(text=f"{default_gain} dB")
        if self.camera is not None:
            self.camera.set_exposure(default_exposure_val)
            self.camera.set_gain(default_gain)
        print("Camera settings reset to defaults.")

    def toggle_preview(self):
        if self.preview_enabled.get():
            self.preview_frame.grid(row=1, column=1, sticky="")
            self.update_preview()
        else:
            self.preview_frame.grid_remove()

    def update_preview(self):
        if self.preview_enabled.get():
            if self.current_frame is not None:
                try:
                    img = Image.fromarray(self.current_frame)
                except Exception:
                    img = Image.new("RGB", (self.preview_width, self.preview_height), (0, 0, 0))
            else:
                img = Image.new("RGB", (self.preview_width, self.preview_height), (0, 0, 0))
            img = ImageOps.pad(img, (self.preview_width, self.preview_height), color=(0, 0, 0))
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
            self.after(100, self.update_preview)

    def update_exposure(self, value):
        try:
            # Compute exposure as 10^(slider value)
            exposure_val = int(10 ** float(value))
            self.exposure_value_label.config(text=f"{exposure_val} us")
            if self.camera is not None:
                self.camera.set_exposure(exposure_val)
                print(f"Exposure set to {exposure_val} us")
        except Exception as e:
            print("Error updating exposure:", e)

    def update_gain(self, value):
        try:
            gain_val = int(float(value))
            self.gain_value_label.config(text=f"{gain_val} dB")
            if self.camera is not None:
                self.camera.set_gain(gain_val)
                print(f"Gain set to {gain_val} dB")
        except Exception as e:
            print("Error updating gain:", e)

    def start_camera(self):
        if not self.running:
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            if self.camera_thread is not None:
                print("camera thread alive:", self.camera_thread.is_alive())
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=False)
            self.camera_thread.start()

    def stop_camera(self):
        if self.running:
            self.running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.camera_thread.join()

    def camera_loop(self):
        try:
            with pyvirtualcam.Camera(width=2448, height=2048, fps=20) as cam:
                print(f"Using virtual camera: {cam.device}")
                while self.running:
                    data_tensor = next(self.camera)

                    # Retrieve the selected view from data_tensor.
                    try:
                        frame = data_tensor[self.selected_view]
                    except Exception as e:
                        print("Error retrieving view:", e)
                        frame = data_tensor["raw"]

                    # If frame is single-channel, convert to 3 channels.
                    if len(frame.shape) == 2:
                        frame = np.stack([frame, frame, frame], axis=-1)
                    elif len(frame.shape) == 3 and frame.shape[-1] == 1:
                        frame = np.repeat(frame, 3, axis=-1)

                    cam.send(frame)
                    cam.sleep_until_next_frame()
                    self.current_frame = frame
                print("Thread ended")
            print("Ended virtualcam")
        except Exception as e:
            print("Error in camera loop:", e)
            self.current_frame = None

    def on_close(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = VirtualCameraApp()
    app.mainloop()
