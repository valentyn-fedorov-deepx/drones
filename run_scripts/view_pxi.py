import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import os
from functools import wraps
from typing import Optional
import sys

from src.data_pixel_tensor import DataPixelTensor


def scale_to_8bit(array):
    # Find the minimum and maximum values
    min_val = array.min()
    max_val = array.max()

    # Scale the array to 0-255
    scaled = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return scaled


def debounce(wait_time):
    """
    Decorator to debounce a function call.

    Args:
        wait_time (float): Time to wait in seconds before allowing another function call
    """
    def decorator(fn):
        last_called = [0]
        timer = [None]

        @wraps(fn)
        def debounced(*args, **kwargs):
            def call_function():
                last_called[0] = time.time()
                fn(*args, **kwargs)

            # Cancel the previous timer if it exists
            if timer[0] is not None:
                args[0].root.after_cancel(timer[0])

            # Set new timer
            elapsed = time.time() - last_called[0]
            if elapsed > wait_time:
                call_function()
            else:
                timer[0] = args[0].root.after(int(wait_time * 1000), call_function)

        return debounced
    return decorator


class ImageProcessorApp:
    def __init__(self, root, pxi_path: Optional[str] = None):
        self.root = root
        self.root.title("RAW Reader")
        self.root.geometry("1800x1600")

        # Variables
        self.file_path = tk.StringVar(value=pxi_path)
        self.channel = tk.StringVar(value="view_img")
        self.contrast = tk.DoubleVar(value=1.0)
        self.brightness = tk.DoubleVar(value=0)
        self.threshold = tk.DoubleVar(value=127)
        self.use_threshold = tk.BooleanVar(value=False)
        self.use_equalization = tk.BooleanVar(value=False)

        # Data storage - cache the DataPixelTensor
        self.data_tensor = None
        self.current_file_path = None

        # State tracking
        self.current_state = {
            'contrast': 1.0,
            'brightness': 0,
            'threshold': 127,
            'use_threshold': False,
            'use_equalization': False,
            'channel': 'view_img'
        }

        # Default values for reset
        self.default_values = self.current_state.copy()

        self.setup_ui()

        # Load initial file if provided
        if pxi_path:
            self.load_data_tensor()

        # Bind window resize event
        self.root.bind('<Configure>', self.handle_resize)
        self.last_window_size = (self.root.winfo_width(),
                                 self.root.winfo_height())

    def setup_ui(self):
        # Main container with grid
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(2, weight=1)

        # Input Frame
        input_frame = ttk.Frame(self.main_container, padding="10")
        input_frame.grid(row=0, column=0, sticky="ew")

        # File Selection
        ttk.Label(input_frame, text="RAW File:").pack(side=tk.LEFT)
        ttk.Entry(input_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)

        # Controls Frame
        controls_frame = ttk.Frame(self.main_container, padding="10")
        controls_frame.grid(row=1, column=0, sticky="ew")

        # Channel Selection
        ttk.Label(controls_frame, text="Channel:").pack(side=tk.LEFT)
        channel_dropdown = ttk.Combobox(controls_frame, textvariable=self.channel,
                                        values=["view_img", "n_z", "n_xy", "n_xyz"],
                                        state="readonly")
        channel_dropdown.pack(side=tk.LEFT, padx=5)

        # Contrast Slider
        ttk.Label(controls_frame, text="Contrast:").pack(side=tk.LEFT, padx=(20, 0))
        contrast_slider = ttk.Scale(controls_frame, from_=0.0, to=3.0, 
                                    variable=self.contrast, orient=tk.HORIZONTAL)
        contrast_slider.pack(side=tk.LEFT, padx=5)

        # Brightness Slider
        ttk.Label(controls_frame, text="Brightness:").pack(side=tk.LEFT, padx=(20, 0))
        brightness_slider = ttk.Scale(controls_frame, from_=-100, to=100,
                                      variable=self.brightness,
                                      orient=tk.HORIZONTAL)
        brightness_slider.pack(side=tk.LEFT, padx=5)

        # Histogram Equalization Checkbox
        self.equalize_checkbox = ttk.Checkbutton(controls_frame,
                                                 text="Autoexposure",
                                                 variable=self.use_equalization)
        self.equalize_checkbox.pack(side=tk.LEFT, padx=(20, 0))

        # Threshold Frame
        self.threshold_frame = ttk.Frame(controls_frame)
        self.threshold_frame.pack(side=tk.LEFT)

        # Threshold Enable Checkbox
        self.threshold_checkbox = ttk.Checkbutton(self.threshold_frame, 
                                                  text="Enable Threshold",
                                                  variable=self.use_threshold,
                                                  command=self.update_threshold_state)
        self.threshold_checkbox.pack(side=tk.LEFT, padx=(20, 0))

        # Threshold Slider
        self.threshold_label = ttk.Label(self.threshold_frame, text="Threshold:")
        self.threshold_label.pack(side=tk.LEFT, padx=(20, 0))
        self.threshold_slider = ttk.Scale(self.threshold_frame, from_=0, to=255,
                                          variable=self.threshold, orient=tk.HORIZONTAL)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        # Buttons Frame
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(side=tk.RIGHT, padx=10)

        # Save Button
        ttk.Button(buttons_frame, text="Save Processed",
                   command=self.save_processed_image).pack(side=tk.RIGHT, padx=5)

        # Reset Button
        ttk.Button(buttons_frame, text="Reset",
                   command=self.reset_controls).pack(side=tk.RIGHT, padx=5)

        # Images Frame with grid
        self.images_frame = ttk.Frame(self.main_container, padding="10")
        self.images_frame.grid(row=2, column=0, sticky="nsew")
        self.images_frame.grid_columnconfigure(0, weight=1)
        self.images_frame.grid_columnconfigure(1, weight=1)
        self.images_frame.grid_rowconfigure(0, weight=1)

        # Original Channel Image
        self.original_label = ttk.Label(self.images_frame)
        self.original_label.grid(row=0, column=0, padx=5, sticky="nsew")

        # Processed Channel Image
        self.processed_label = ttk.Label(self.images_frame)
        self.processed_label.grid(row=0, column=1, padx=5, sticky="nsew")

        # Initially update threshold visibility
        self.update_threshold_visibility()

        # Bind events with debouncing
        self.file_path.trace_add("write", self.on_file_path_change)
        self.channel.trace_add("write", self.on_channel_change)

        # Bind slider events
        for var, name in [(self.contrast, 'contrast'),
                          (self.brightness, 'brightness'),
                          (self.threshold, 'threshold')]:
            var.trace_add("write", lambda *args,
                          n=name: self.handle_value_change(n))

        # Bind checkbox events
        for var, name in [(self.use_threshold, 'use_threshold'),
                          (self.use_equalization, 'use_equalization')]:
            var.trace_add("write", lambda *args, n=name: self.handle_value_change(n))

    @debounce(0.05)
    def handle_resize(self, event):
        """Handle window resize events with debouncing"""
        if event.widget == self.root:
            current_size = (event.width, event.height)
            if current_size != self.last_window_size:
                self.last_window_size = current_size
                self.update_images(force=True)

    def handle_value_change(self, param_name):
        """Handle value changes and determine if update is needed"""
        new_value = getattr(self, param_name).get()
        if new_value != self.current_state[param_name]:
            self.current_state[param_name] = new_value
            self.debounced_update()

    @debounce(0.05)
    def debounced_update(self):
        """Debounced version of update_images"""
        self.update_images()

    def on_file_path_change(self, *args):
        """Handle file path changes - load new DataPixelTensor"""
        new_path = self.file_path.get()
        if new_path != self.current_file_path:
            self.current_file_path = new_path
            if new_path:  # Only load if path is not empty
                self.load_data_tensor()
            self.update_images(force=True)

    def browse_file(self):
        """Open file dialog for selecting PXI files"""
        file_path = filedialog.askopenfilename(filetypes=[("PXIs", '*.pxi')])
        if file_path:
            self.file_path.set(file_path)

    def load_data_tensor(self):
        """Load DataPixelTensor from the current file path"""
        try:
            if self.file_path.get():
                print(f"Loading DataPixelTensor from: {self.file_path.get()}")
                self.data_tensor = DataPixelTensor.from_file(
                    self.file_path.get(),
                    lazy_calculations=True,
                    color_data=True
                )
                print("DataPixelTensor loaded successfully")
            else:
                self.data_tensor = None
        except Exception as e:
            print(f"Error loading DataPixelTensor: {e}")
            self.data_tensor = None

    def reset_controls(self):
        """Reset all controls to their default values"""
        self.contrast.set(self.default_values['contrast'])
        self.brightness.set(self.default_values['brightness'])
        self.threshold.set(self.default_values['threshold'])
        self.use_threshold.set(self.default_values['use_threshold'])
        self.use_equalization.set(self.default_values['use_equalization'])

    def update_threshold_state(self):
        """Update the threshold slider state based on checkbox"""
        state = "normal" if self.use_threshold.get() else "disabled"
        self.threshold_slider.configure(state=state)
        self.update_images()

    def update_threshold_visibility(self):
        """Update the visibility of threshold controls based on channel selection"""
        if self.channel.get() == "z":
            self.threshold_frame.pack(side=tk.LEFT)
            self.update_threshold_state()
        else:
            self.threshold_frame.pack_forget()

    def on_channel_change(self, *args):
        """Handle channel change events"""
        new_channel = self.channel.get()
        if new_channel != self.current_state['channel']:
            self.current_state['channel'] = new_channel
            self.update_threshold_visibility()
            self.update_images(force=True)

    def get_current_image(self):
        """Get the current image data from the cached DataPixelTensor"""
        if self.data_tensor is None:
            return None

        try:
            element = getattr(self.data_tensor, self.channel.get())
            if element.dtype == np.uint16:
                element = scale_to_8bit(element)
            return element
        except AttributeError:
            print(f"Channel '{self.channel.get()}' not found in DataPixelTensor")
            return None

    def equalize_image(self, image):
        """Apply histogram equalization to image"""
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            equalized = np.zeros_like(image)
            for i in range(image.shape[2]):
                equalized[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return equalized

    def calculate_image_size(self):
        """Calculate the target size for images based on window size"""
        frame_width = self.images_frame.winfo_width()
        frame_height = self.images_frame.winfo_height()

        target_width = (frame_width - 30) // 2
        target_height = frame_height - 20

        return (target_width, target_height)

    def save_processed_image(self):
        """Save the processed image as PNG"""
        if not hasattr(self, 'processed_photo') or not self.file_path.get():
            print("No processed image to save")
            return

        try:
            original_path = self.file_path.get()
            base_path = os.path.splitext(original_path)[0]
            processed_path = f"{base_path}_processed.png"

            if not hasattr(self, 'processed_image_data'):
                print("No processed image data available")
                return

            image_to_save = (self.processed_image_data.astype(np.uint8)
                             if self.processed_image_data.dtype != np.uint8
                             else self.processed_image_data)

            Image.fromarray(image_to_save).save(processed_path)
            print(f"Saved processed image to: {processed_path}")

        except Exception as e:
            print(f"Error saving processed image: {e}")

    def update_images(self, force=False):
        """Update images with optional force parameter"""
        if not self.file_path.get() or self.data_tensor is None:
            return

        try:
            # Get the current image from cached DataPixelTensor
            image = self.get_current_image()
            if image is None:
                print("Failed to get image data")
                return

            processed = image.copy().astype(float)

            if self.use_equalization.get():
                processed_uint8 = np.clip(processed, 0, 255).astype(np.uint8)
                processed = self.equalize_image(processed_uint8).astype(float)

            processed = processed * self.contrast.get()
            processed = processed + self.brightness.get()

            if self.channel.get() == "z" and self.use_threshold.get():
                processed = np.where(processed > self.threshold.get(), 255, 0)

            processed = np.clip(processed, 0, 255).astype(np.uint8)
            self.processed_image_data = processed.copy()

            self.update_image_display(image, processed)

        except Exception as e:
            print(f"Error processing image: {e}")

    def update_image_display(self, original, processed):
        """Handle the UI update portion of image processing"""
        if original.dtype == np.uint16:
            original = scale_to_8bit(original)

        if processed.dtype == np.uint16:
            processed = scale_to_8bit(processed)

        original_img = Image.fromarray(original)
        processed_img = Image.fromarray(processed)

        target_size = self.calculate_image_size()
        if target_size[0] > 0 and target_size[1] > 0:
            aspect_ratio = original_img.width / original_img.height

            if aspect_ratio > 1:
                new_width = min(target_size[0], int(target_size[1] * aspect_ratio))
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(target_size[1], int(target_size[0] / aspect_ratio))
                new_width = int(new_height * aspect_ratio)

            # Resize images
            original_img = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            processed_img = processed_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.original_photo = ImageTk.PhotoImage(original_img)
        self.processed_photo = ImageTk.PhotoImage(processed_img)

        # Update labels
        self.original_label.configure(image=self.original_photo)
        self.processed_label.configure(image=self.processed_photo)


def main(pxi_path: Optional[str] = None):
    root = tk.Tk()
    app = ImageProcessorApp(root, pxi_path)
    root.mainloop()


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = None

    main(data_path)