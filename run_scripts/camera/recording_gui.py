import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
from loguru import logger
import sys
import math

from src.camera import LiveCamera, CUSTOM_AUTOEXPOSURE_METHODS
from src.threads.raw_recorder import DataRecorder
from src.threads.capture_thread import CaptureThread
from src.data_pixel_tensor import DataPixelTensor
from configs.data_pixel_tensor import DISPLAY_NAMES 
from deployment.remote_camera_management.request_models import SetCameraParamsRequest
from src.utils.ridge_map import label_ridges
from src.utils.raw_processing import DATA_PIXEL_TENSOR_BACKEND


class CameraGUI:
    def __init__(self, master, save_path=None, device_idx=0):
        self.master = master
        self.master.title("Camera Control Interface")
        self.master.geometry("1200x800")

        # Initialize camera and recording variables
        self.device_idx = device_idx
        self.save_path = Path(save_path) if save_path else Path("./recordings")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Camera and recording state
        self.live_camera = None
        self.recording = False
        self.recording_thread = None
        self.data_recorder = None
        self.capture_thread_processor = None
        self.capture_thread = None

        # Threading objects
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_capturing_event = threading.Event()
        self.stop_saving_event = threading.Event()
        self.camera_params_change_queue = queue.Queue(maxsize=5)

        # GUI variables
        self.current_frame = None
        self.photo = None
        self.roi_xyxyn = None  # To store normalized ROI (x1, y1, x2, y2)
        self.roi_start_x = None
        self.roi_start_y = None
        self.current_roi_rect_id = None
        self.image_display_info = {} # To store image display params

        # Exposure logarithmic scale variables
        self.exposure_min = 1
        self.exposure_max = 1000000  # Will be updated when camera is initialized
        self.exposure_slider_resolution = 1000  # Number of steps in the slider

        # Initialize GUI
        self.setup_gui()
        self.initialize_camera()

        # Start GUI update loop
        self.update_display()

    def log_to_linear(self, log_value, min_val, max_val, resolution):
        """Convert logarithmic value to linear slider position"""
        if log_value <= 0:
            return 0
        log_min = math.log10(max(min_val, 1))
        log_max = math.log10(max_val)
        log_val = math.log10(log_value)

        # Normalize to 0-1 range
        normalized = (log_val - log_min) / (log_max - log_min)
        # Scale to slider resolution
        return int(normalized * resolution)

    def linear_to_log(self, linear_value, min_val, max_val, resolution):
        """Convert linear slider position to logarithmic value"""
        if linear_value <= 0:
            return min_val

        log_min = math.log10(max(min_val, 1))
        log_max = math.log10(max_val)

        # Normalize linear value to 0-1 range
        normalized = linear_value / resolution
        # Convert to log scale
        log_val = log_min + normalized * (log_max - log_min)
        return int(10 ** log_val)

    def on_exposure_scale_change(self, value):
        """Handle exposure slider change and update display"""
        linear_value = float(value)
        actual_exposure = self.linear_to_log(linear_value, self.exposure_min,
                                             self.exposure_max, self.exposure_slider_resolution)
        self.exposure_var.set(str(actual_exposure))

    def on_gain_scale_change(self, value):
        """Handle gain slider change and update display"""
        self.gain_var.set(str(int(float(value))))

    def on_half_kernel_size_change(self, value):
        """Handle half kernel size slider change and update display"""
        self.half_kernel_size_var.set(str(int(float(value))))

    def on_nz_threshold_change(self, value):
        """Handle n_z threshold slider change and update display"""
        self.nz_threshold_var.set(f"{float(value):.3f}")

    def on_custom_autoexposure_change(self, event=None):
        """Handle custom autoexposure mode selection."""
        if not self.live_camera:
            messagebox.showwarning("Warning", "Camera not initialized")
            return

        selected_key = self.custom_autoexposure_mode_var.get()
        if not selected_key:
            return
        try:
            if hasattr(self.live_camera, 'set_custom_autoexposure'):
                self.live_camera.set_custom_autoexposure(CUSTOM_AUTOEXPOSURE_METHODS[selected_key])
                # Optionally, provide feedback to the user
                logger.info(f"Custom AutoExposure mode set to {selected_key}")
            else:
                messagebox.showerror("Error", "Feature not implemented in LiveCamera: 'set_custom_autoexposure'")
                logger.error("LiveCamera instance does not have 'set_custom_autoexposure' method.")
        except Exception as e:
            logger.error(f"Failed to set custom autoexposure mode: {e}")
            messagebox.showerror("Error", f"Failed to set custom autoexposure mode: {e}")

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Camera Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Camera initialization
        ttk.Label(control_frame, text="Device Index:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar(value=str(self.device_idx))
        device_entry = ttk.Entry(control_frame, textvariable=self.device_var, width=10)
        device_entry.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Button(control_frame, text="Initialize Camera",
                   command=self.initialize_camera).grid(row=0, column=2, padx=(10, 0), pady=2)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=1, column=0, columnspan=3,
                                                              sticky=(tk.W, tk.E), pady=10)

        # Camera parameters
        ttk.Label(control_frame, text="Camera Parameters",
                  font=('TkDefaultFont', 10, 'bold')).grid(row=2, column=0, columnspan=3, pady=(0, 10))

        # Exposure (logarithmic scale)
        ttk.Label(control_frame, text="Exposure:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.exposure_var = tk.StringVar(value=100)
        self.exposure_scale = ttk.Scale(control_frame, from_=0, to=self.exposure_slider_resolution,
                                        orient=tk.HORIZONTAL, command=self.on_exposure_scale_change)
        self.exposure_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        self.exposure_label = ttk.Label(control_frame, textvariable=self.exposure_var, width=8)
        self.exposure_label.grid(row=3, column=2, padx=(5, 0), pady=2)

        # Gain (integer scale)
        ttk.Label(control_frame, text="Gain:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.gain_var = tk.StringVar()
        self.gain_scale = ttk.Scale(control_frame, from_=0, to=100,
                                    orient=tk.HORIZONTAL,
                                    command=self.on_gain_scale_change)
        self.gain_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=2)
        self.gain_label = ttk.Label(control_frame, textvariable=self.gain_var, width=8)
        self.gain_label.grid(row=4, column=2, padx=(5, 0), pady=2)

        # Pixel Format
        ttk.Label(control_frame, text="Pixel Format:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.pixel_format_var = tk.StringVar()
        self.pixel_format_combo = ttk.Combobox(control_frame, textvariable=self.pixel_format_var,
                                              state="readonly", width=15)
        self.pixel_format_combo.grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Exposure Mode
        ttk.Label(control_frame, text="Exposure Mode:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.exposure_mode_var = tk.StringVar()
        self.exposure_mode_combo = ttk.Combobox(control_frame, textvariable=self.exposure_mode_var,
                                                state="readonly", width=15)
        self.exposure_mode_combo.grid(row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Custom AutoExposure Mode
        ttk.Label(control_frame, text="Custom AutoExposure Mode:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.custom_autoexposure_mode_var = tk.StringVar()
        self.custom_autoexposure_mode_combo = ttk.Combobox(control_frame, 
                                                           textvariable=self.custom_autoexposure_mode_var,
                                                           state="readonly", width=15)
        self.custom_autoexposure_mode_combo['values'] = list(CUSTOM_AUTOEXPOSURE_METHODS.keys())

        self.custom_autoexposure_mode_combo.set("None")
        self.custom_autoexposure_mode_combo.grid(row=7, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        # self.custom_autoexposure_mode_combo.bind("<<ComboboxSelected>>", self.on_custom_autoexposure_change)

        # Apply parameters button
        ttk.Button(control_frame, text="Apply Parameters",
                   command=self.apply_camera_params).grid(row=8, column=0, columnspan=3, pady=10)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=3,
                                                               sticky=(tk.W, tk.E), pady=10)

        # Recording controls
        ttk.Label(control_frame, text="Recording Controls",
                 font=('TkDefaultFont', 10, 'bold')).grid(row=10, column=0, columnspan=3, pady=(0, 10))

        # Save path
        ttk.Label(control_frame, text="Save Path:").grid(row=11, column=0, sticky=tk.W, pady=2)
        self.save_path_var = tk.StringVar(value=str(self.save_path))
        save_path_entry = ttk.Entry(control_frame, textvariable=self.save_path_var, width=20)
        save_path_entry.grid(row=11, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(control_frame, text="Browse",
                  command=self.browse_save_path).grid(row=11, column=2, padx=(5, 0), pady=2)

        # Recording name
        ttk.Label(control_frame, text="Recording Name:").grid(row=12, column=0, sticky=tk.W, pady=2)
        self.record_name_var = tk.StringVar(value=f"recording_{int(time.time())}")
        record_name_entry = ttk.Entry(control_frame, textvariable=self.record_name_var, width=20)
        record_name_entry.grid(row=12, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Recording buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=13, column=0, columnspan=3, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Recording",
                                      command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="Stop Recording",
                                     command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        # Status information
        ttk.Separator(control_frame, orient='horizontal').grid(row=14, column=0, columnspan=3,
                                                              sticky=(tk.W, tk.E), pady=10)

        ttk.Label(control_frame, text="Status",
                 font=('TkDefaultFont', 10, 'bold')).grid(row=15, column=0, columnspan=3, pady=(0, 10))

        # Recording status
        self.recording_status_var = tk.StringVar(value="Not Recording")
        ttk.Label(control_frame, text="Recording:").grid(row=16, column=0, sticky=tk.W, pady=2)
        self.recording_status_label = ttk.Label(control_frame, textvariable=self.recording_status_var)
        self.recording_status_label.grid(row=16, column=1, columnspan=2, sticky=tk.W, pady=2)

        # Frame count
        self.frame_count_var = tk.StringVar(value="0")
        ttk.Label(control_frame, text="Frames Saved:").grid(row=17, column=0, sticky=tk.W, pady=2)
        ttk.Label(control_frame, textvariable=self.frame_count_var).grid(row=17, column=1, columnspan=2, sticky=tk.W, pady=2)

        # Camera info
        self.camera_info_var = tk.StringVar(value="Camera not initialized")
        ttk.Label(control_frame, text="Camera Info:").grid(row=18, column=0, sticky=tk.W, pady=2)
        info_label = ttk.Label(control_frame, textvariable=self.camera_info_var, wraplength=200)
        info_label.grid(row=18, column=1, columnspan=2, sticky=tk.W, pady=2)

        # Ridge Map Parameters
        ttk.Separator(control_frame, orient='horizontal').grid(row=19, column=0, columnspan=3,
                                                              sticky=(tk.W, tk.E), pady=10)

        ttk.Label(control_frame, text="Ridge Map Parameters",
                  font=('TkDefaultFont', 10, 'bold')).grid(row=20, column=0, columnspan=3, pady=(0, 10))

        # Half Kernel Size slider
        ttk.Label(control_frame, text="Half Kernel Size:").grid(row=21, column=0, sticky=tk.W, pady=2)
        self.half_kernel_size_var = tk.StringVar(value="2")
        self.half_kernel_size_scale = ttk.Scale(control_frame, from_=2, to=50,
                                               orient=tk.HORIZONTAL,
                                               command=self.on_half_kernel_size_change)
        self.half_kernel_size_scale.grid(row=21, column=1, sticky=(tk.W, tk.E), pady=2)
        self.half_kernel_size_scale.set(2)
        self.half_kernel_size_label = ttk.Label(control_frame, textvariable=self.half_kernel_size_var, width=8)
        self.half_kernel_size_label.grid(row=21, column=2, padx=(5, 0), pady=2)

        # n_z Threshold checkbox
        self.enable_nz_threshold_var = tk.BooleanVar(value=False)
        self.nz_threshold_checkbox = ttk.Checkbutton(control_frame,
                                                    text="Enable n_z Threshold",
                                                    variable=self.enable_nz_threshold_var)
        self.nz_threshold_checkbox.grid(row=22, column=0, columnspan=3, sticky=tk.W, pady=2)

        # n_z Threshold slider
        ttk.Label(control_frame, text="n_z Threshold:").grid(row=23, column=0, sticky=tk.W, pady=2)
        self.nz_threshold_var = tk.StringVar(value="0.500")
        self.nz_threshold_scale = ttk.Scale(control_frame, from_=0.0, to=1.0,
                                           orient=tk.HORIZONTAL,
                                           command=self.on_nz_threshold_change)
        self.nz_threshold_scale.grid(row=23, column=1, sticky=(tk.W, tk.E), pady=2)
        self.nz_threshold_scale.set(0.5)
        self.nz_threshold_label = ttk.Label(control_frame, textvariable=self.nz_threshold_var, width=8)
        self.nz_threshold_label.grid(row=23, column=2, padx=(5, 0), pady=2)

        # Right panel - Video display
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=640, height=480)
        self.video_canvas.pack(expand=True, fill=tk.BOTH)

        # Display mode selection
        display_frame = ttk.Frame(video_frame)
        display_frame.pack(pady=(10, 0))

        ttk.Label(display_frame, text="Display Mode:").pack(side=tk.LEFT)
        self.display_mode_var = tk.StringVar(value="raw")
        display_combo = ttk.Combobox(display_frame, textvariable=self.display_mode_var,
                                     values=DISPLAY_NAMES + ['ridgemap'], state="readonly", width=10)
        display_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # ROI Checkbox
        self.enable_roi_var = tk.BooleanVar(value=False)
        self.roi_checkbox = ttk.Checkbutton(display_frame,
                                            text="Enable ROI",
                                            variable=self.enable_roi_var,
                                            command=self.toggle_roi_mode)
        self.roi_checkbox.pack(side=tk.LEFT)

        # Configure column weights for control frame
        control_frame.columnconfigure(1, weight=1)

    def toggle_roi_mode(self):
        """Enable or disable ROI selection mode."""
        if self.enable_roi_var.get():
            # Bind mouse events for drawing
            self.video_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
            self.video_canvas.bind("<B1-Motion>", self.on_canvas_drag)
            self.video_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
            if self.live_camera._custom_autoexposure_active:
                self.live_camera._custom_autoexposure.set_roi_from_xyxyn(self.roi_xyxyn)
        else:
            # Unbind mouse events
            self.video_canvas.unbind("<ButtonPress-1>")
            self.video_canvas.unbind("<B1-Motion>")
            self.video_canvas.unbind("<ButtonRelease-1>")
            # Remove the ROI rectangle from the canvas
            self.video_canvas.delete("roi_rect")
            if self.live_camera._custom_autoexposure_active:
                self.live_camera._custom_autoexposure.set_roi_from_xyxyn(None)

    def on_canvas_press(self, event):
        """Handle the start of ROI drawing."""
        self.roi_start_x = event.x
        self.roi_start_y = event.y
        # Delete any existing rectangle
        self.video_canvas.delete("roi_rect")

    def on_canvas_drag(self, event):
        """Handle dragging to draw the ROI rectangle for visual feedback."""
        if self.roi_start_x is None or self.roi_start_y is None:
            return
        
        # Delete the old rectangle before drawing a new one
        if self.current_roi_rect_id:
            self.video_canvas.delete(self.current_roi_rect_id)

        # Draw a new one
        self.current_roi_rect_id = self.video_canvas.create_rectangle(
            self.roi_start_x, self.roi_start_y, event.x, event.y,
            outline='red', width=2, tags="roi_rect_feedback"
        )

    def on_canvas_release(self, event):
        """Handle the end of ROI drawing and normalize coordinates."""
        if self.roi_start_x is None or self.roi_start_y is None:
            return

        # Clean up the feedback rectangle
        self.video_canvas.delete("roi_rect_feedback")
        self.current_roi_rect_id = None

        # Final coordinates
        end_x, end_y = event.x, event.y

        # Get image display parameters to perform normalization
        img_info = self.image_display_info
        if not img_info:
            logger.warning("Cannot set ROI, image display info not available.")
            return

        img_w, img_h = img_info['width'], img_info['height']
        offset_x, offset_y = img_info['x_offset'], img_info['y_offset']

        # Convert canvas coordinates to image coordinates
        x1_img = self.roi_start_x - offset_x
        y1_img = self.roi_start_y - offset_y
        x2_img = end_x - offset_x
        y2_img = end_y - offset_y

        # Clamp coordinates to be within the image bounds
        x1_img = max(0, min(x1_img, img_w))
        y1_img = max(0, min(y1_img, img_h))
        x2_img = max(0, min(x2_img, img_w))
        y2_img = max(0, min(y2_img, img_h))

        # Ensure x1 < x2 and y1 < y2 for the final normalized tuple
        norm_x1 = min(x1_img, x2_img) / img_w
        norm_y1 = min(y1_img, y2_img) / img_h
        norm_x2 = max(x1_img, x2_img) / img_w
        norm_y2 = max(y1_img, y2_img) / img_h

        # Store the normalized ROI
        self.roi_xyxyn = (norm_x1, norm_y1, norm_x2, norm_y2)
        logger.info(f"New ROI selected (normalized xyxyn): {self.roi_xyxyn}")

        if self.live_camera._custom_autoexposure_active:
            self.live_camera._custom_autoexposure.set_roi_from_xyxyn(self.roi_xyxyn)

        # Reset start coordinates and draw the final box
        self.roi_start_x = None
        self.roi_start_y = None
        self.draw_roi_box()

    def draw_roi_box(self):
        """Draws the stored ROI box on the canvas, scaled to the current image display."""
        # First, clear any existing ROI box to prevent duplicates
        self.video_canvas.delete("roi_rect")

        if not self.roi_xyxyn or not self.enable_roi_var.get():
            return

        img_info = self.image_display_info
        if not img_info: # Cannot draw if we don't know the image geometry
            return

        img_w, img_h = img_info['width'], img_info['height']
        offset_x, offset_y = img_info['x_offset'], img_info['y_offset']

        # De-normalize coordinates from 0-1 range to canvas pixel coordinates
        norm_x1, norm_y1, norm_x2, norm_y2 = self.roi_xyxyn
        x1 = norm_x1 * img_w + offset_x
        y1 = norm_y1 * img_h + offset_y
        x2 = norm_x2 * img_w + offset_x
        y2 = norm_y2 * img_h + offset_y

        # Draw the final rectangle
        self.video_canvas.create_rectangle(x1, y1, x2, y2, outline='cyan', width=2, tags="roi_rect")


    def browse_save_path(self):
        """Browse for save directory"""
        path = filedialog.askdirectory(initialdir=self.save_path)
        if path:
            self.save_path_var.set(path)
            self.save_path = Path(path)

    def initialize_camera(self):
        """Initialize the camera and start capture thread"""
        try:
            device_idx = int(self.device_var.get())

            # Stop existing camera if running
            if self.capture_thread and self.capture_thread.is_alive():
                self.stop_capturing_event.set()
                self.capture_thread.join()

            # Initialize camera
            self.live_camera = LiveCamera(device_idx, lazy_calculations=True,
                                          return_data_tensor=True)

            # Update camera info
            self.update_camera_info()

            # Reset events
            self.stop_capturing_event.clear()

            # Start capture thread
            self.capture_thread_processor = CaptureThread(self.live_camera,
                                                          self.frame_queue,
                                                          self.stop_capturing_event,
                                                          self.camera_params_change_queue)
            self.capture_thread = threading.Thread(target=self.capture_thread_processor.capture_loop,
                                                   daemon=True)
            self.capture_thread.start()

            messagebox.showinfo("Success", f"Camera {device_idx} initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            messagebox.showerror("Error", f"Failed to initialize camera: {e}")

    def update_exposure_gain_display_info(self):
        # Get exposure and gain ranges
        exposure_range = self.live_camera.exposure_range
        gain_range = self.live_camera.gain_range

        info = f"Exposure: {exposure_range[0]}-{exposure_range[1]}\n"
        info += f"Gain: {gain_range[0]}-{gain_range[1]}"
        self.camera_info_var.set(info)

        # Update exposure range for logarithmic scale
        self.exposure_min = max(exposure_range[0], 1)  # Ensure minimum is at least 1 for log scale
        self.exposure_max = exposure_range[1]

        # Update gain scale range (integer)
        self.gain_scale.configure(from_=gain_range[0], to=gain_range[1])

        # Set current values from camera properties
        try:
            # Get current exposure and convert to slider position
            current_exposure = getattr(self.live_camera, 'current_exposure', None)
            if current_exposure is not None and current_exposure > 0:
                current_exposure = int(current_exposure)
                slider_pos = self.log_to_linear(current_exposure, self.exposure_min,
                                                self.exposure_max, self.exposure_slider_resolution)
                self.exposure_scale.set(slider_pos)
                self.exposure_var.set(str(current_exposure))
            else:
                self.exposure_scale.set(0)
                self.exposure_var.set(str(self.exposure_min))

            # Get current gain
            current_gain = getattr(self.live_camera, 'current_gain', None)
            if current_gain is not None:
                current_gain = int(current_gain)
                self.gain_scale.set(current_gain)
                self.gain_var.set(str(current_gain))
            else:
                self.gain_scale.set(gain_range[0])
                self.gain_var.set(str(gain_range[0]))

        except Exception as e:
            logger.warning(f"Could not read current exposure/gain: {e}")
            # Set to minimum values if current values not available
            self.exposure_scale.set(0)
            self.exposure_var.set(str(self.exposure_min))
            self.gain_scale.set(gain_range[0])
            self.gain_var.set(str(int(gain_range[0])))


    def update_camera_info(self):
        """Update camera information display"""
        if self.live_camera:
            try:
                self.update_exposure_gain_display_info()

                # Update pixel format dropdown
                try:
                    available_formats = self.live_camera.available_pixel_formats
                    format_names = list(available_formats.keys())
                    self.pixel_format_combo.configure(values=format_names)

                    # Set current pixel format
                    current_pixel_format = getattr(self.live_camera, 'current_pixel_format', None)
                    if current_pixel_format and current_pixel_format in format_names:
                        self.pixel_format_var.set(current_pixel_format)
                    elif format_names:
                        # Set first format as default if current not available or not in list
                        self.pixel_format_var.set(format_names[0])

                except Exception as e:
                    logger.warning(f"Could not update pixel format: {e}")

                # Update exposure mode dropdown
                try:
                    exposure_modes = self.live_camera.available_exposure_modes
                    exposure_mode_names = list(exposure_modes.keys())
                    self.exposure_mode_combo.configure(values=exposure_mode_names)

                    # Set current exposure mode
                    current_exposure_mode = getattr(self.live_camera, 'current_exposure_mode', None)
                    if current_exposure_mode and current_exposure_mode in exposure_mode_names:
                        self.exposure_mode_var.set(current_exposure_mode)
                    elif exposure_mode_names:
                        # Set first mode as default if current not available or not in list
                        self.exposure_mode_var.set(exposure_mode_names[0])

                except Exception as e:
                    logger.warning(f"Could not update exposure mode: {e}")

            except AttributeError as e:
                logger.warning(f"Some camera properties not available: {e}")
                self.camera_info_var.set("Limited camera info available")
        else:
            self.camera_info_var.set("Camera not initialized")

    def apply_camera_params(self):
        """Apply camera parameters"""
        if not self.live_camera:
            messagebox.showwarning("Warning", "Camera not initialized")
            return

        try:
            # Create parameter request
            params = {}

            # Add exposure if changed (convert from display value)
            if self.exposure_var.get():
                params['exposure'] = int(self.exposure_var.get())  # Use the actual exposure value, not slider position

            # Add gain if changed (already an integer)
            if self.gain_var.get():
                params['gain'] = int(self.gain_var.get())

            # Add pixel format if selected
            selected_format = self.pixel_format_var.get()
            if selected_format:
                available_formats = self.live_camera.available_pixel_formats
                if selected_format in available_formats:
                    params['pixel_format'] = selected_format

            # Add exposure mode if selected
            selected_exposure_mode = self.exposure_mode_var.get()
            if selected_exposure_mode:
                exposure_modes = self.live_camera.available_exposure_modes
                if selected_exposure_mode in exposure_modes:
                    params['exposure_mode'] = selected_exposure_mode

            selected_custom_exposure_mode = self.custom_autoexposure_mode_var.get()
            if selected_custom_exposure_mode:
                if selected_custom_exposure_mode in CUSTOM_AUTOEXPOSURE_METHODS:
                    params['custom_exposure_mode'] = selected_custom_exposure_mode

            if params:
                # Put parameters in queue for the capture thread to process
                try:
                    param_request = SetCameraParamsRequest(**params)
                    self.camera_params_change_queue.put_nowait(param_request)
                    messagebox.showinfo("Success", f"Camera parameters applied: {list(params.keys())}")
                except queue.Full:
                    messagebox.showwarning("Warning", "Parameter queue is full")
            else:
                messagebox.showwarning("Warning", "No parameters to apply")

        except Exception as e:
            logger.error(f"Failed to apply camera parameters: {e}")
            messagebox.showerror("Error", f"Failed to apply camera parameters: {e}")

    def start_recording(self):
        """Start recording"""
        if self.recording:
            messagebox.showwarning("Warning", "Recording already in progress")
            return

        if not self.live_camera:
            messagebox.showwarning("Warning", "Camera not initialized")
            return

        try:
            # Setup recording path
            record_name = self.record_name_var.get() or f"recording_{int(time.time())}"
            self.save_path = Path(self.save_path_var.get())
            record_path = self.save_path / record_name
            record_path.mkdir(parents=True, exist_ok=True)

            # Initialize data recorder
            self.stop_saving_event.clear()
            self.data_recorder = DataRecorder(record_path, self.frame_queue,
                                              self.stop_saving_event)

            # Start recording thread
            self.recording = True
            self.recording_thread = threading.Thread(target=self.data_recorder.start_recording_loop,
                                                     args=(".pxi",), daemon=True)
            self.recording_thread.start()

            # Update UI
            self.recording_status_var.set("Recording")
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)

            messagebox.showinfo("Success", f"Recording started: {record_name}")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            messagebox.showerror("Error", f"Failed to start recording: {e}")

    def stop_recording(self):
        """Stop recording"""
        if not self.recording:
            messagebox.showwarning("Warning", "Recording is not active")
            return

        try:
            self.recording = False
            self.stop_saving_event.set()

            if self.recording_thread:
                self.recording_thread.join(timeout=5)

            self.data_recorder = None

            # Update UI
            self.recording_status_var.set("Not Recording")
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)

            messagebox.showinfo("Success", "Recording stopped")

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            messagebox.showerror("Error", f"Failed to stop recording: {e}")

    def update_display(self):
        """Update the video display and status"""
        try:
            # Update frame display
            if hasattr(self, 'capture_thread_processor') and self.capture_thread_processor:
                latest_data = self.capture_thread_processor.latest_data
                if latest_data is not None:
                    self.display_frame(latest_data)

            # Update frame count
            if self.recording and self.data_recorder:
                count = getattr(self.data_recorder, 'current_save_idx', 0)
                self.frame_count_var.set(str(count))

        except Exception as e:
            logger.error(f"Error updating display: {e}")

        # self.update_exposure_gain_display_info()

        # Schedule next update
        self.master.after(33, self.update_display)  # ~30 FPS

    def apply_nz_threshold(self, data_representation):
        """Apply threshold to n_z data if enabled"""
        if self.enable_nz_threshold_var.get():
            threshold_value = float(self.nz_threshold_var.get())
            # Apply threshold: values below threshold become 0, above become 1
            data_representation = data_representation / data_representation.max()
            thresholded = (data_representation > threshold_value).astype(np.uint8) * 255
            return thresholded
        return data_representation

    def display_frame(self, data_tensor):
        """Display frame on canvas"""
        try:
            # Convert data to displayable format
            if isinstance(data_tensor, np.ndarray):
                data_tensor = DataPixelTensor(data_tensor, color_data=self.live_camera.color_data,
                                              lazy_calculations=self.live_camera.lazy_calculations,
                                              for_display=True)

            # Get the requested representation
            element = self.display_mode_var.get()
            if element == 'ridgemap':
                # Get the half kernel size from the slider
                half_kernel_size = int(self.half_kernel_size_var.get())
                if DATA_PIXEL_TENSOR_BACKEND == 'torch':
                    view_img = data_tensor.view_img.cpu().numpy()
                else:
                    view_img = data_tensor.view_img
                data_representation = label_ridges(view_img, half_kernel_size=half_kernel_size)
            else:
                data_representation = data_tensor[element]

                if DATA_PIXEL_TENSOR_BACKEND == 'torch' and element != 'raw':
                    data_representation = data_representation.cpu().numpy()

            # Apply n_z threshold if this is n_z data and threshold is enabled
            if element == 'n_z':
                data_representation = self.apply_nz_threshold(data_representation)

            # Convert to uint8 if needed
            if data_representation.dtype == np.uint16:
                data_representation = data_representation.astype(float) / data_tensor.max_value
                data_representation = (data_representation * 255).astype(np.uint8)
            elif data_representation.dtype == np.float32 or data_representation.dtype == np.float64:
                # Handle float data (including thresholded data)
                data_representation = np.clip(data_representation, 0, 1)
                data_representation = (data_representation * 255).astype(np.uint8)

            # Convert grayscale to RGB if needed
            if len(data_representation.shape) == 2:
                data_representation = cv2.cvtColor(data_representation, cv2.COLOR_GRAY2RGB)

            # Convert to PIL Image
            image = Image.fromarray(data_representation)

            # Resize to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
                # Maintain aspect ratio
                img_width, img_height = image.size
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(image)

                # Clear canvas and display image
                self.video_canvas.delete("all")
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2
                self.video_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)

                # Store image display parameters for ROI calculation
                self.image_display_info = {
                    'width': new_width,
                    'height': new_height,
                    'x_offset': x_offset,
                    'y_offset': y_offset
                }

                # Draw the ROI box if enabled and it exists
                self.draw_roi_box()


        except Exception as e:
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = traceback.extract_tb(exc_traceback)
            line_number = tb[-1].lineno
            filename = tb[-1].filename

            print(f"Exception occurred: {e}")
            print(f"File: {filename}")
            print(f"Line: {line_number}")
            logger.error(f"Error displaying frame: {e}")

    def on_closing(self):
        """Handle window closing"""
        try:
            # Stop recording if active
            if self.recording:
                self.stop_recording()

            # Stop capture thread
            if hasattr(self, 'stop_capturing_event'):
                self.stop_capturing_event.set()

            if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        self.master.destroy()


def main():
    """Main function to run the GUI"""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--save-path", type=Path, default="./recordings")
    parser.add_argument("--device-idx", default=0, type=int)
    parser.add_argument("--log-level", default="DEBUG")

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Create GUI
    root = tk.Tk()
    app = CameraGUI(root, save_path=args.save_path, device_idx=args.device_idx)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()