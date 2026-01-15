import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import os
from functools import wraps
from typing import Optional
from loguru import logger

from src.data_pixel_tensor import DataPixelTensor
from src.offline_utils.frame_source import FrameSource
from configs.data_pixel_tensor import DATA_PIXEL_TENSOR_BACKEND


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def scale_to_8bit(array: np.ndarray) -> np.ndarray:
    """Scales a numpy array to 8-bit (0-255) for display."""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val == min_val:
        return np.zeros_like(array, dtype=np.uint8)

    # Scale the array to 0-255
    scaled = (255.0 * (array.astype(np.float32) - min_val) / (max_val - min_val))
    return scaled.astype(np.uint8)


# ==============================================================================
# MAIN APPLICATION CLASS
# ==============================================================================

class FolderProcessorApp:
    def __init__(self, root: tk.Tk, data_path: Optional[str] = None):
        self.root = root
        self.root.title("Image Sequence Processor")
        self.root.geometry("1800x1000")

        # --- State and Data Variables ---
        self.source_path = tk.StringVar(value=data_path or "")
        self.channel = tk.StringVar(value="view_img")
        self.contrast = tk.DoubleVar(value=1.0)
        self.brightness = tk.DoubleVar(value=0)
        self.use_equalization = tk.BooleanVar(value=False)
        self.frame_position = tk.IntVar(value=0)
        self.is_playing = tk.BooleanVar(value=False)
        
        self.frame_source: Optional[FrameSource] = None
        self.current_frame_data: Optional[DataPixelTensor] = None
        self.processed_image_data: Optional[np.ndarray] = None
        
        # --- UI Setup ---
        self.setup_ui()
        
        # --- Initial Load ---
        if data_path:
            self.load_frame_source()

        # --- Bindings ---
        self.source_path.trace_add("write", lambda *args: self.load_frame_source())
        self.channel.trace_add("write", lambda *args: self.update_display())
        self.contrast.trace_add("write", lambda *args: self.update_display())
        self.brightness.trace_add("write", lambda *args: self.update_display())
        self.use_equalization.trace_add("write", lambda *args: self.update_display())
        self.frame_position.trace_add("write", lambda *args: self.on_frame_select())
        self.root.bind('<Configure>', self.on_resize)

    def setup_ui(self):
        self.main_container = ttk.Frame(self.root, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(3, weight=1) # Add weight to image row

        # --- Row 0: Source Path ---
        source_frame = ttk.Frame(self.main_container)
        source_frame.grid(row=0, column=0, sticky="ew", pady=5)
        ttk.Label(source_frame, text="Source Folder:").pack(side=tk.LEFT)
        ttk.Entry(source_frame, textvariable=self.source_path, width=70).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(source_frame, text="Browse...", command=self.browse_folder).pack(side=tk.LEFT)

        # --- Row 1: Image Controls ---
        controls_frame = ttk.Frame(self.main_container)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=5)
        # Channel
        ttk.Label(controls_frame, text="Channel:").pack(side=tk.LEFT)
        ttk.Combobox(controls_frame, textvariable=self.channel, values=["view_img", "n_z", "n_xy", "n_xyz"], state="readonly").pack(side=tk.LEFT, padx=5)
        # Contrast
        ttk.Label(controls_frame, text="Contrast:").pack(side=tk.LEFT, padx=(15, 0))
        ttk.Scale(controls_frame, from_=0.0, to=3.0, variable=self.contrast, orient=tk.HORIZONTAL).pack(side=tk.LEFT, padx=5)
        # Brightness
        ttk.Label(controls_frame, text="Brightness:").pack(side=tk.LEFT, padx=(15, 0))
        ttk.Scale(controls_frame, from_=-100, to=100, variable=self.brightness, orient=tk.HORIZONTAL).pack(side=tk.LEFT, padx=5)
        # Autoexposure
        ttk.Checkbutton(controls_frame, text="Autoexposure", variable=self.use_equalization).pack(side=tk.LEFT, padx=(20, 0))
        # Save Button
        ttk.Button(controls_frame, text="Save Processed", command=self.save_processed_image).pack(side=tk.RIGHT, padx=5)


        # --- Row 2: Playback Controls ---
        playback_frame = ttk.Frame(self.main_container)
        playback_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.play_button = ttk.Button(playback_frame, text="▶ Start", command=self.toggle_playback, width=10)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.frame_label = ttk.Label(playback_frame, text="Frame: 0 / 0", width=15)
        self.frame_label.pack(side=tk.LEFT)

        self.frame_slider = ttk.Scale(playback_frame, from_=0, to=0, variable=self.frame_position, orient=tk.HORIZONTAL)
        self.frame_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)


        # --- Row 3: Image Display ---
        self.images_frame = ttk.Frame(self.main_container)
        self.images_frame.grid(row=3, column=0, sticky="nsew", pady=10)
        self.images_frame.grid_columnconfigure(0, weight=1)
        self.images_frame.grid_columnconfigure(1, weight=1)
        self.images_frame.grid_rowconfigure(0, weight=1)

        self.original_label = ttk.Label(self.images_frame, text="Original Image", anchor="center")
        self.original_label.grid(row=0, column=0, padx=5, sticky="nsew")
        self.processed_label = ttk.Label(self.images_frame, text="Processed Image", anchor="center")
        self.processed_label.grid(row=0, column=1, padx=5, sticky="nsew")

    def browse_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            self.source_path.set(folder_path)

    def load_frame_source(self):
        path = self.source_path.get()
        if not path or not os.path.isdir(path):
            self.frame_source = None
            self.clear_displays()
            return
        
        try:
            self.frame_source = FrameSource(path)
            if len(self.frame_source) == 0:
                logger.warning(f"No supported image files found in {path}")
                self.frame_source = None
                self.clear_displays()
                return
            
            # Update UI for the new source
            self.frame_slider.config(to=len(self.frame_source) - 1)
            self.frame_position.set(0) # This will trigger on_frame_select
            self.is_playing.set(False)
            self.play_button.config(text="▶ Start")
            
        except Exception as e:
            logger.error(f"Error initializing FrameSource: {e}")
            self.frame_source = None
            self.clear_displays()
    
    def on_frame_select(self):
        if not self.frame_source:
            return
        
        pos = self.frame_position.get()
        self.frame_label.config(text=f"Frame: {pos + 1} / {len(self.frame_source)}")
        
        try:
            self.current_frame_data = self.frame_source[pos]
            self.update_display(force_resize=True)
        except (IndexError, IOError) as e:
            logger.error(f"Could not load frame at position {pos}: {e}")
            self.clear_displays()

    def update_display(self, force_resize=False):
        if not self.current_frame_data:
            return
        
        # --- Image Processing Logic ---
        try:
            original = self.current_frame_data[self.channel.get()]
            if original is None:
                return
            
            if DATA_PIXEL_TENSOR_BACKEND == 'torch':
                original = original.cpu().numpy()

            # Handle high bit-depth images for processing
            if original.dtype != np.uint8:
                processed = scale_to_8bit(original).astype(np.float32)
            else:
                processed = original.copy().astype(np.float32)

            # Apply enhancements
            if self.use_equalization.get():
                img_uint8 = np.clip(processed, 0, 255).astype(np.uint8)
                if len(img_uint8.shape) == 2: # Grayscale
                    processed = cv2.equalizeHist(img_uint8).astype(np.float32)
                else: # Color
                    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
                    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                    processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB).astype(np.float32)

            processed = processed * self.contrast.get() + self.brightness.get()
            self.processed_image_data = np.clip(processed, 0, 255).astype(np.uint8)

            self.update_image_widgets(original, self.processed_image_data, force_resize)

        except Exception as e:
            logger.error(f"Error processing image: {e}")

    def update_image_widgets(self, original: np.ndarray, processed: np.ndarray, force: bool = False):
        if original.dtype != np.uint8:
            original = scale_to_8bit(original)
        
        # Calculate target size
        frame_width = self.images_frame.winfo_width()
        frame_height = self.images_frame.winfo_height()
        if frame_width < 50 or frame_height < 50: return # Avoid resizing when window is minimized
        
        target_w = (frame_width - 30) // 2
        target_h = frame_height - 20
        
        aspect_ratio = original.shape[1] / original.shape[0]
        
        new_w = min(target_w, int(target_h * aspect_ratio))
        new_h = int(new_w / aspect_ratio)
        
        if new_w <= 0 or new_h <= 0: return

        # Create PhotoImage objects
        orig_img = Image.fromarray(original).resize((new_w, new_h), Image.Resampling.LANCZOS)
        proc_img = Image.fromarray(processed).resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.original_photo = ImageTk.PhotoImage(orig_img)
        self.processed_photo = ImageTk.PhotoImage(proc_img)

        self.original_label.config(image=self.original_photo)
        self.processed_label.config(image=self.processed_photo)

    def on_resize(self, event):
        # Update display on resize to fit images correctly
        if self.current_frame_data:
            self.update_display(force_resize=True)
            
    def toggle_playback(self):
        if not self.frame_source:
            return
        
        if self.is_playing.get():
            self.is_playing.set(False)
            self.play_button.config(text="▶ Start")
        else:
            self.is_playing.set(True)
            self.play_button.config(text="❚❚ Stop")
            self.play_next_frame()

    def play_next_frame(self):
        if not self.is_playing.get() or not self.frame_source:
            self.is_playing.set(False)
            self.play_button.config(text="▶ Start")
            return
            
        current_pos = self.frame_position.get()
        next_pos = current_pos + 1
        
        if next_pos >= len(self.frame_source):
            self.is_playing.set(False) # Stop at the end
            self.play_button.config(text="▶ Start")
            return
            
        self.frame_position.set(next_pos) # This triggers the update
        
        # Schedule the next frame
        delay = int(1000 / self.frame_source.fps)
        self.root.after(delay, self.play_next_frame)

    def save_processed_image(self):
        if self.processed_image_data is None or not self.frame_source:
            logger.warning("No processed image to save.")
            return
            
        try:
            current_pos = self.frame_position.get()
            original_path = self.frame_source.files_list[current_pos]
            
            base_name = original_path.stem
            output_dir = original_path.parent / "processed"
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f"{base_name}_processed.png"
            
            Image.fromarray(self.processed_image_data).save(output_path)
            logger.info(f"Saved processed image to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            
    def clear_displays(self):
        self.original_label.config(image='', text="Load a folder to begin")
        self.processed_label.config(image='', text="")
        self.frame_label.config(text="Frame: 0 / 0")
        self.frame_slider.config(to=0)
        self.frame_position.set(0)


def main(data_path: Optional[str] = None):
    root = tk.Tk()
    app = FolderProcessorApp(root, data_path)
    root.mainloop()


if __name__ == "__main__":
    import sys
    # You can provide a path to a folder of images as a command-line argument
    # e.g., python your_script_name.py /path/to/image_folder
    folder_to_load = sys.argv[1] if len(sys.argv) > 1 else None
    main(folder_to_load)