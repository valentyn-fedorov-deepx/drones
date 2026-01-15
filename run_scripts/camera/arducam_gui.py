import os
import argparse
import subprocess
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import socket
from pathlib import Path

from loguru import logger
import tinytuya
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.camera.arducam_200mp import Camera, CameraXU
from src.utils.raw_processing.utils_raw_processing_numpy import gray_world_white_balance

# --- Configuration Constants ---
# Grouping configuration at the top makes the script easier to manage.
VIDEO_CAPTURE_APIS = {
    "ANY": cv2.CAP_ANY,
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
    "V4L2": cv2.CAP_V4L2,
}

BAYER_COLOR_ORDERS = {
    "BG": cv2.COLOR_BayerBG2BGR,
    "GB": cv2.COLOR_BayerGB2BGR,
    "RG": cv2.COLOR_BayerRG2BGR,
    "GR": cv2.COLOR_BayerGR2BGR,
}

color_order_list = [
    cv2.COLOR_BayerGB2BGR,
    cv2.COLOR_BayerBG2BGR,
    cv2.COLOR_BayerRG2BGR,
    cv2.COLOR_BayerGR2BGR,
]

LASER_DEVICE_ID = "eb71db492577d91ca6cqxz"
LASER_LOCAL_KEY = "yH>e@-6Sb@5#R:l&"

LED_DEVICE_ID = "ebc39efe00d9470594mzdr"
LED_LOCAL_KEY = "dT&Pr&wUf;tN<1<X"


# 200MP Conversion Configuration
HIGH_RES_WIDTH = 16320
HIGH_RES_HEIGHT = 12288
INF_EEPROM_PATH = "inf_eeprom.dat"
MAC_EEPROM_PATH = "mac_eeprom.dat"


def downscale_to_mp(image: Image.Image) -> Image.Image:
    """
    Downscales a PIL Image to a maximum of 5 megapixels, preserving the aspect ratio.

    Args:
        image: The input PIL Image object.

    Returns:
        The downscaled PIL Image object. If the original image is already
        5 megapixels or smaller, the original image is returned.
    """
    # Define the target resolution in megapixels
    TARGET_MEGAPIXELS = 0.8
    TARGET_PIXELS = TARGET_MEGAPIXELS * 1_000_000

    # Get the original image dimensions
    width, height = image.size
    current_pixels = width * height

    # Check if downscaling is necessary
    if current_pixels <= TARGET_PIXELS:
        return image

    scale_factor = np.sqrt(TARGET_PIXELS / current_pixels)

    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image using the LANCZOS filter for high-quality downscaling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_image

def convert_200mp(filepath, color_order=0):
    width = 16320
    height = 12288
    # convert_exe_path = r"./200mp_convert_lib\arducam_200mp_convert.exe"
    convert_exe_path = r".\src\camera\arducam_lib\arducam_200mp_convert.exe"
    inf_eeprom_data_path = "inf_eeprom.dat"
    mac_eeprom_data_path = "mac_eeprom.dat"
    eeprom_data_len = 4608

    basename = os.path.splitext(filepath)[0]
    image_data_raw8 = np.fromfile(filepath, dtype=np.uint8)
    image_data_raw10 = (image_data_raw8.astype(np.uint16) << 2)
    out_raw_path = f"{basename}_raw10"
    image_data_raw10.tofile(f"{out_raw_path}.raw")
    command = [
        convert_exe_path,
        out_raw_path,
        str(width), str(height), str(width), str(height), str(color_order),
        inf_eeprom_data_path, str(eeprom_data_len),
        mac_eeprom_data_path, str(eeprom_data_len),
    ]

    try:
        logger.debug("Executing command: {}".format(" ".join(command)))
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        end_time = time.time()
        logger.debug("Command executed successfully!")
        logger.debug("Output:")
        logger.debug(result.stdout)
        logger.debug(f"Conversion time: {end_time - start_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while executing the command: {e.stderr}")
        return -1
    
    out_raw_path = "out_normal.raw"
    data = np.fromfile(out_raw_path, dtype=np.uint8)
    arr = np.frombuffer(data, dtype=np.uint16)
    arr = arr >> 2
    arr = arr.astype(np.uint8)
    image = arr.reshape(height, width, 1)
    image = cv2.cvtColor(image, color_order_list[color_order])
    return image

class CameraApp:
    def __init__(self, master: tk.Tk, args: argparse.Namespace):
        self.master = master
        self.args = args

        # --- Camera & App State ---
        self.cap = None
        self.current_frame = None
        self.save_dir = Path.cwd() / "captures"
        self.camera_xu = CameraXU()
        self.camera_names = []
        self.camera_names_real = []
        self.current_index = 0
        self.autofocus_var = tk.BooleanVar()

        # --- Tuya Relay State ---
        self.laser_relay = None
        self.led_relay = None

        # --- FPS Counter Attributes (Encapsulated in the class) ---
        self.fps_start_time = time.monotonic()
        self.frame_count = 0

        # --- ROI Selection Attributes ---
        self.use_roi_var = tk.BooleanVar(value=False)
        self.roi_xyxyn = None  # Stores final ROI as [x_min, y_min, x_max, y_max] normalized
        self.roi_start_pos = None # Stores (x, y) of mouse press
        self.roi_preview_rect = None # Stores the current preview rectangle coordinates

        self.setup_ui()
        self.update_relays() # Initialize Tuya relays
        self.refresh_cameras()  # Initial camera scan and open
        self.update_frame()


    def update_relays(self):
        """
        Initializes Tuya relay devices with error handling.
        Sets device attributes to None if initialization fails.
        """
        # --- Initialize Laser Relay ---
        try:
            logger.debug("Connecting to Laser Relay...")
            self.laser_relay = tinytuya.OutletDevice(
                dev_id=LASER_DEVICE_ID,
                address=None, # Using IP for speed
                local_key=LASER_LOCAL_KEY
            )
            self.laser_relay.set_version(3.4)
            # A quick status check confirms connection
            _ = self.laser_relay.status() 
            logger.debug("Laser Relay connected successfully.")
        except (socket.timeout, ConnectionRefusedError, ConnectionError, Exception) as e:
            self.laser_relay = None
            messagebox.showwarning(
                "Relay Connection Error",
                f"Could not connect to the Laser Relay.\nError: {e}\n\nCheck device power, network, and configuration."
            )
            logger.error(f"Error connecting to Laser Relay: {e}")

        # --- Initialize LED Relay ---
        try:
            logger.debug("Connecting to LED Relay...")
            self.led_relay = tinytuya.OutletDevice(
                dev_id=LED_DEVICE_ID,
                address=None, # Using IP for speed
                local_key=LED_LOCAL_KEY
            )
            self.led_relay.set_version(3.4)
            # A quick status check confirms connection
            _ = self.led_relay.status()
            logger.debug("LED Relay connected successfully.")
        except (socket.timeout, ConnectionRefusedError, ConnectionError, Exception) as e:
            self.led_relay = None
            messagebox.showwarning(
                "Relay Connection Error",
                f"Could not connect to the LED Relay.\nError: {e}\n\nCheck device power, network, and configuration."
            )
            logger.error(f"Error connecting to LED Relay: {e}")
        

    def init_camera(self, index: int):
        if self.cap and self.cap.isOpened():
            self.cap.release()

        if not self.camera_names:
            messagebox.showerror("Error", "No cameras found.")
            return

        logger.debug(f"Opening camera: {self.camera_names[index]} ({index})")
        # Use dictionary to look up the API constant
        # self.cap = Camera(index, cv2.CAP_ANY)
        self.cap = Camera(index, cv2.CAP_DSHOW)
        self.cap.open()
        self.cap.set_width(self.args.width)
        self.cap.set_height(self.args.height)
        self.cap.set_fps(self.args.FrameRate)

        # Set autofocus and manual exposure
        if self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1):
            logger.debug("Autofocus enabled.")
        else:
            logger.warning("Warning: Autofocus not supported.")

        if self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25): # 0.25 is manual mode
            logger.debug("Manual exposure mode set.")
            # Set initial exposure if provided
            if self.args.Exposure is not None:
                self.exp_scale.set(self.args.Exposure)
                self.on_exposure_change(self.args.Exposure)
        else:
            logger.warning("[yellow]Warning: Could not set manual exposure.[/yellow]")

        # Set initial focus if provided
        if self.args.Focus is not None:
            self.focus_scale.set(self.args.Focus)
            self.on_focus_change(self.args.Focus)

    def setup_ui(self):
        self.master.title("Camera Preview")

        # --- Main Window Grid Configuration ---
        # Column 0: Controls panel (fixed width)
        # Column 1: Video display (takes remaining space)
        self.master.columnconfigure(0, weight=0)
        self.master.columnconfigure(1, weight=1)
        # Make the main row expand vertically
        self.master.rowconfigure(0, weight=1)

        # --- Left Panel for All Controls ---
        left_panel = ttk.Frame(self.master)
        left_panel.grid(row=0, column=0, padx=10, pady=5, sticky="ns")

        # --- Video Display (Right Column) ---
        self.video_label = ttk.Label(self.master)
        self.video_label.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # --- ROI Mouse Bindings ---
        self.video_label.bind("<ButtonPress-1>", self.on_mouse_press)
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_release)

        # --- Controls Frame (Inside Left Panel) ---
        controls_frame = ttk.LabelFrame(left_panel, text="Controls")
        controls_frame.pack(fill="x", expand=True, pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        # Camera selection
        ttk.Label(controls_frame, text="Camera:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.cam_var = tk.StringVar()
        self.cam_menu = ttk.Combobox(controls_frame, textvariable=self.cam_var, state='readonly')
        self.cam_menu.grid(row=0, column=1, sticky='ew', padx=5)
        self.cam_menu.bind('<<ComboboxSelected>>', self.on_camera_change)
        ttk.Button(controls_frame, text="Refresh", command=self.refresh_cameras).grid(row=0, column=2, padx=5)

        # Save Directory
        ttk.Label(controls_frame, text="Save To:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.dir_entry = ttk.Entry(controls_frame)
        self.dir_entry.insert(0, str(self.save_dir))
        self.dir_entry.grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Button(controls_frame, text="Browse...", command=self.browse_directory).grid(row=1, column=2, padx=5)

        # Focus Slider
        ttk.Label(controls_frame, text="Focus:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.focus_scale = ttk.Scale(controls_frame, from_=0, to=1023, command=self.on_focus_change)
        self.focus_scale.grid(row=2, column=1, columnspan=2, sticky='ew', padx=5)

        # Exposure Slider
        ttk.Label(controls_frame, text="Exposure:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.exp_scale = ttk.Scale(controls_frame, from_=-11, to=-1, command=self.on_exposure_change)
        self.exp_scale.grid(row=3, column=1, columnspan=2, sticky='ew', padx=5)

        # ROI and Autofocus Checkboxes
        self.roi_check = ttk.Checkbutton(controls_frame, text="Use ROI", variable=self.use_roi_var)
        self.roi_check.grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.autofocus_check = ttk.Checkbutton(
            controls_frame,
            text="Automatic Focus",
            variable=self.autofocus_var,
            command=self.toggle_custom_autofocus
        )
        self.autofocus_check.grid(row=4, column=1, columnspan=2, sticky='w', padx=5, pady=2)

        # --- Project U Controls Frame (Inside Left Panel) ---
        project_u_frame = ttk.LabelFrame(left_panel, text="Project U Controls")
        project_u_frame.pack(fill="x", expand=True, pady=10)

        # Variables for the checkboxes
        self.switch1_var, self.switch2_var, self.switch3_var, self.switch4_var, self.switch5_var = (tk.BooleanVar() for _ in range(5))

        # Define switches with their text, variable, and target relay/switch number
        # Format: (Text, Variable, (Relay_Name, Switch_Index))
        switches = [
            ("Laser 1", self.switch1_var, ("laser", 1)),
            ("LED 1", self.switch2_var, ("led", 1)),
            ("LED 2", self.switch3_var, ("led", 2)),
            ("LED 3", self.switch4_var, ("led", 3)),
            ("LED 4", self.switch5_var, ("led", 4)),
        ]

        # Grid layout for switches
        for i, (text, var, relay_info) in enumerate(switches):
            # The lambda captures the relay_info and var for the command
            cmd = lambda r=relay_info, v=var: self.toggle_relay_switch(r, v)
            ttk.Checkbutton(
                project_u_frame, text=text, variable=var, command=cmd
            ).grid(row=i // 3, column=i % 3, sticky='w', padx=5, pady=2)

        # Center the button below the switches
        for i in range(3):
            project_u_frame.columnconfigure(i, weight=1)
        ttk.Button(project_u_frame, text="Start Project U capture", command=self.start_project_u_capture).grid(row=2, column=0, columnspan=3, pady=5)


        # --- Action Buttons (Inside Left Panel) ---
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save Preview", command=self.save_preview_frame).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Capture 200MP", command=self.capture_highres).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Quit", command=self.quit).pack(side='left', padx=5)

    def toggle_relay_switch(self, relay_info: tuple, state_var: tk.BooleanVar):
        """Toggles a specific switch on a Tuya relay with error handling."""
        relay_name, switch_index = relay_info
        is_on = state_var.get()
        state_text = "ON" if is_on else "OFF"
    
        target_relay = None
        if relay_name == "laser":
            target_relay = self.laser_relay
        elif relay_name == "led":
            target_relay = self.led_relay
    
        # 1. Check if the relay object was initialized successfully
        if target_relay is None:
            logger.debug(f"Cannot toggle {relay_name.capitalize()} Relay: device is not connected.")
            # Silently fail
            return
    
        logger.debug(f"Attempting to turn {relay_name.capitalize()} Switch {switch_index} -> {state_text}")
    
        # 2. Try to send the command, catching potential communication errors
        try:
            target_relay.set_status(is_on, switch=switch_index)
            logger.debug(f"Successfully turned {relay_name.capitalize()} Switch {switch_index} {state_text}.")
        except Exception as e:
            messagebox.showerror(
                "Relay Command Error",
                f"Failed to set status for {relay_name.capitalize()} Relay.\nError: {e}\n\nThe device may have gone offline."
            )
            logger.error(f"Error sending command to {relay_name.capitalize()} Relay: {e}")
            # Revert the checkbox state since the command failed
            state_var.set(not is_on)


    def start_project_u_capture(self):
        """Runs the capture sequence, ensuring all required devices are online first."""
        logger.info("[bold green]Starting Project U capture...[/bold green]")
        
        # --- Pre-flight Check ---
        if not self.laser_relay or not self.led_relay:
            messagebox.showerror(
                "Capture Aborted",
                "Cannot start Project U capture because one or more Tuya relays are offline. Please check connections and restart."
            )
            logger.error("Aborting capture: Relays not available.")
            return

        messagebox.showinfo("Project U", "Capture sequence initiated!")
        wait_time = 2

        laser_switches = [1, 2]
        for laser_switch_idx in laser_switches:
            self.laser_relay.set_status(False, switch=laser_switch_idx)

        led_switches = [1, 2, 3, 4]
        for led_switch_idx in led_switches:
            self.led_relay.set_status(False, switch=led_switch_idx)
        
        logger.debug("Switched off everything")
        directory_name = time.strftime('project_u_%Y-%m-%d_%H_%M_%S')
        project_u_logs_path = self.save_dir / directory_name
        project_u_logs_path.mkdir(parents=True, exist_ok=True)

        time.sleep(wait_time)

        for i in range(3):
            ret, black_frame = next(self.cap)

        black_frame_save_path = project_u_logs_path / "black.png"
        cv2.imwrite(str(black_frame_save_path), black_frame)

        logger.debug("Saved black frame")

        laser_switches = [1, 2]
        for laser_switch_idx in laser_switches:
            self.laser_relay.set_status(True, switch=laser_switch_idx)
        
        logger.debug("Turned on laser")
        time.sleep(wait_time*3)
        for i in range(3):
            ret, laser_led_off_frame = next(self.cap)

        laser_led_off_frame_save_path = project_u_logs_path / "laser_led_off.png"
        cv2.imwrite(str(laser_led_off_frame_save_path), laser_led_off_frame)

        logger.debug("Saved laser image with leds off")

        laser_switches = [1, 2]
        for laser_switch_idx in laser_switches:
            self.laser_relay.set_status(False, switch=laser_switch_idx)

        time.sleep(wait_time * 2)

        led_frames_save_path = project_u_logs_path / "original_led"
        led_frames_save_path.mkdir(exist_ok=True)

        led_frames_subtracted_save_path = project_u_logs_path / "black_subtracked"
        led_frames_subtracted_save_path.mkdir(exist_ok=True)

        for led_switch_idx in led_switches:
            self.led_relay.set_status(True, switch=led_switch_idx)
            time.sleep(wait_time*1.5)
            for i in range(5):
                ret, led_frame = next(self.cap)
            
            black_subtracted = led_frame.astype(int) - black_frame
            black_subtracted = np.clip(black_subtracted, 0, 255).astype(np.uint8)

            save_led_frame_subtracted_path = led_frames_subtracted_save_path / f"led{led_switch_idx}.png"
            cv2.imwrite(str(save_led_frame_subtracted_path), black_subtracted)

            save_led_frame_path = led_frames_save_path / f"led{led_switch_idx}.png"
            cv2.imwrite(str(save_led_frame_path), led_frame)

            self.led_relay.set_status(False, switch=led_switch_idx)
            time.sleep(wait_time)
            logger.debug("Saved image with one led")
        
        for laser_switch_idx in laser_switches:
            self.laser_relay.set_status(True, switch=laser_switch_idx)
        
        logger.debug("Turned everything on")
        time.sleep(wait_time*2)

        for led_switch_idx in led_switches:
            self.led_relay.set_status(True, switch=led_switch_idx)
        time.sleep(wait_time*2)

        for i in range(3):
            ret, laser_led_on_frame = next(self.cap)
        laser_led_on_frame_save_path = project_u_logs_path / "laser_led_on.png"
        cv2.imwrite(str(laser_led_on_frame_save_path), laser_led_on_frame)

        logger.debug("Turned everything on and save laser image with leds on")

        for laser_switch_idx in laser_switches:
            self.laser_relay.set_status(False, switch=laser_switch_idx)

        for led_switch_idx in led_switches:
            self.led_relay.set_status(False, switch=led_switch_idx)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = next(self.cap)
            if ret:
                # Use a copy of the frame for drawing so the original is preserved
                frame_to_display = frame.copy()
                self.current_frame = frame

                self._display_fps()

                # --- Draw ROI Rectangle if enabled and defined ---
                if self.use_roi_var.get():
                    # Get original frame dimensions
                    frame_h, frame_w, _ = frame_to_display.shape

                    try:
                        # Get the dimensions of the image currently in the Tkinter label
                        displayed_w = self.video_label.imgtk.width()
                        displayed_h = self.video_label.imgtk.height()

                        # Ensure dimensions are valid before trying to draw
                        if displayed_w > 0 and displayed_h > 0:

                            # Draw the BLUE rectangle being actively dragged (the preview)
                            if self.roi_preview_rect:
                                # Calculate scaling factors from display-space back to frame-space
                                scale_x = frame_w / displayed_w
                                scale_y = frame_h / displayed_h

                                # Scale the preview coordinates to the original frame's size
                                x1 = int(self.roi_preview_rect[0] * scale_x)
                                y1 = int(self.roi_preview_rect[1] * scale_y)
                                x2 = int(self.roi_preview_rect[2] * scale_x)
                                y2 = int(self.roi_preview_rect[3] * scale_y)

                                # Draw the correctly scaled rectangle on the frame
                                cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for active drawing

                            # Draw the final, saved GREEN ROI
                            elif self.roi_xyxyn:
                                x1 = int(self.roi_xyxyn[0] * frame_w)
                                y1 = int(self.roi_xyxyn[1] * frame_h)
                                x2 = int(self.roi_xyxyn[2] * frame_w)
                                y2 = int(self.roi_xyxyn[3] * frame_h)
                                cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for saved

                    except AttributeError:
                        # This handles the very first frame before self.video_label.imgtk exists.
                        pass

                # Convert the (now modified) frame for Tkinter display
                img = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = downscale_to_mp(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            else:
                self.handle_restart()

        # Schedule the next frame update
        self.master.after(int(1000 / self.args.FrameRate), self.update_frame)

    def _display_fps(self):
        """ Internal method to calculate and print FPS to the console. """
        self.frame_count += 1
        now = time.monotonic()
        elapsed = now - self.fps_start_time
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            w, h = self.current_frame.shape[1], self.current_frame.shape[0]
            self.frame_count = 0
            self.fps_start_time = now

    def _get_and_validate_save_dir(self) -> Path | None:
        """
        DRY Principle: Single method to get, validate, and create the save directory.
        Returns a Path object on success, None on failure.
        """
        try:
            dir_path = Path(self.dir_entry.get()).expanduser().resolve()
            dir_path.mkdir(parents=True, exist_ok=True)
            self.save_dir = dir_path # Update the class attribute
            return dir_path
        except Exception as e:
            messagebox.showerror("Invalid Directory", f"Could not use or create the directory:\n{self.dir_entry.get()}\n\nError: {e}")
            return None

    # --- UI Event Handlers ---
    def refresh_cameras(self):
        """
        Scans for available cameras and populates the dropdown with generic names.
        The actual camera identifiers are stored internally.
        """
        logger.info("Refreshing camera list...")
        # Get the actual camera names/identifiers from the hardware library
        self.camera_names_real = self.camera_xu.refresh()

        # --- MODIFIED LINE ---
        # Create a list of generic names for the user interface (e.g., "Camera 0", "Camera 1")
        # The underscore `_` is a convention to indicate that we are ignoring the actual name value.
        self.camera_names = [f"Camera {i}" for i, _ in enumerate(self.camera_names_real)]

        # Update the UI dropdown with the new generic list
        self.cam_menu['values'] = self.camera_names

        if self.camera_names:
            # Set the selection to the first camera and initialize it
            self.cam_var.set(self.camera_names[0])
            self.on_camera_change()
        else:
            logger.error("No cameras found.")

    def on_camera_change(self, event=None):
        try:
            idx = self.cam_menu.current()
            if idx != self.current_index or not self.cap:
                self.current_index = idx
                self.init_camera(idx)
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to switch camera: {e}")

    def browse_directory(self):
        dir_ = filedialog.askdirectory(initialdir=str(self.save_dir))
        if dir_:
            self.save_dir = Path(dir_)
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, str(self.save_dir))

    def on_focus_change(self, val):
        self.cap.set_focus(int(float(val)))

    def on_exposure_change(self, val):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(val))

    def on_mouse_press(self, event):
        """ Handles the start of an ROI selection. """
        if not self.use_roi_var.get() or self.current_frame is None:
            return
        # Reset any previous ROI
        self.roi_xyxyn = None
        self.roi_start_pos = (event.x, event.y)
        # Initialize the preview rectangle
        self.roi_preview_rect = (event.x, event.y, event.x, event.y)

    def on_mouse_drag(self, event):
        """ Handles dragging to define the ROI rectangle. """
        if not self.use_roi_var.get() or self.roi_start_pos is None:
            return
        # Update the preview rectangle coordinates
        x0, y0 = self.roi_start_pos
        self.roi_preview_rect = (x0, y0, event.x, event.y)

    def on_mouse_release(self, event):
        """ Handles the end of an ROI selection, calculating and storing the final coordinates. """
        if not self.use_roi_var.get() or self.roi_start_pos is None or self.current_frame is None:
            return

        # --- Finalize preview rect and clear drawing states ---
        x_start, y_start = self.roi_start_pos
        x_end, y_end = event.x, event.y
        self.roi_start_pos = None
        self.roi_preview_rect = None # Stop drawing the temporary blue box

        # --- Get necessary dimensions ---
        # Get dimensions of the original, full-resolution frame
        frame_h, frame_w, _ = self.current_frame.shape

        # Get the actual dimensions of the image displayed in the label.
        # This is the key fix: use the real size of the displayed image.
        try:
            displayed_w = self.video_label.imgtk.width()
            displayed_h = self.video_label.imgtk.height()
        except AttributeError:
            # This can happen if the first frame hasn't been rendered yet.
            logger.error("Error: Could not determine displayed image size to set ROI.")
            return

        # Avoid division by zero if image has no size
        if displayed_w == 0 or displayed_h == 0:
            return

        # --- Convert mouse coordinates to normalized frame coordinates ---
        # The coordinates are normalized against the displayed image dimensions.
        # Since the aspect ratio is preserved, these normalized coordinates are
        # directly applicable to the original full-resolution frame.
        norm_x1 = min(x_start, x_end) / displayed_w
        norm_y1 = min(y_start, y_end) / displayed_h
        norm_x2 = max(x_start, x_end) / displayed_w
        norm_y2 = max(y_start, y_end) / displayed_h

        # Save the final, clipped, normalized coordinates (xyxyn format)
        self.roi_xyxyn = [
            np.clip(norm_x1, 0.0, 1.0),
            np.clip(norm_y1, 0.0, 1.0),
            np.clip(norm_x2, 0.0, 1.0),
            np.clip(norm_y2, 0.0, 1.0)
        ]

        logger.debug(f"ROI selected (xyxyn): {self.roi_xyxyn}")
        if self.cap._custom_autofocus_active:
            self.cap._custom_autofocus.set_roi_from_xyxyn(self.roi_xyxyn)

    def toggle_custom_autofocus(self):
        """Toggles the custom autofocus feature."""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Autofocus", "Camera not initialized.")
            self.autofocus_var.set(False)  # Reset checkbox
            return

        try:
            if self.autofocus_var.get():
                logger.info("Starting custom autofocus...")
                self.cap.set_custom_autofocus()
                self.focus_scale.configure(state='disabled')  # Disable manual focus
                if self.roi_xyxyn is not None:
                    self.cap._custom_autofocus.set_roi_from_xyxyn(self.roi_xyxyn)
            else:
                logger.info("Stopping custom autofocus...")
                self.cap.turn_off_custom_autofocus()
                self.focus_scale.configure(state='normal')  # Re-enable manual focus
        except AttributeError:
            messagebox.showerror("Autofocus Error", "This camera does not support the custom autofocus method.")
            self.autofocus_var.set(False)
            self.focus_scale.configure(state='normal')
        except Exception as e:
            messagebox.showerror("Autofocus Error", f"An error occurred while toggling autofocus:\n{e}")
            self.autofocus_var.set(False)
            self.focus_scale.configure(state='normal')

    def handle_restart(self):
        for i in range(self.args.reStartTimes):
            logger.info(f"Camera stream lost. Attempting restart {i+1}/{self.args.reStartTimes}...")
            try:
                self.cap.reStart()
                logger.info("[green]Camera restart successful.[/green]")
                return
            except Exception as e:
                logger.warning(f"Restart attempt failed: {e}")
                time.sleep(0.5)
        logger.error("[red]All camera restart attempts failed.[/red]")

    # --- Main Actions ---
    def save_preview_frame(self):
        if self.current_frame is None:
            messagebox.showwarning("Save", "No frame available to save.")
            return

        save_dir = self._get_and_validate_save_dir()
        if not save_dir:
            return

        fname = time.strftime(f'{self.args.width}x{self.args.height}_%Y%m%d_%H%M%S.png')
        file_path = save_dir / fname

        try:
            # Note: The saved frame will include the green ROI rectangle if it's visible
            cv2.imwrite(str(file_path), self.current_frame)
            logger.info(f"Preview saved: {file_path}")
            messagebox.showinfo("Save", f"Image saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to write image to:\n{file_path}\n\nError: {e}")

    def capture_highres(self):
        # ——— read & validate the directory from the UI, exactly as in save_frame() ———
        try:
            dir_path = Path(self.dir_entry.get()).expanduser().resolve()
        except Exception as e:
            messagebox.showerror("Save Error", f"Invalid path:\n{e}")
            return
        if dir_path.exists() and not dir_path.is_dir():
            messagebox.showerror("Save Error", f"Path exists but is not a folder:\n{dir_path}")
            return
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not create folder:\n{dir_path}\n\n{e}")
            return

        # finally update internal save_dir so everything else follows suit
        self.save_dir = dir_path

        # ——— now do the high-res capture as before ———
        orig_w, orig_h, orig_fps = self.args.width, self.args.height, self.args.FrameRate


        self.cap.release()
        time.sleep(0.5)
        self.cap = Camera(self.current_index, cv2.CAP_ANY)
        time.sleep(0.5)
        self.cap.open()
        time.sleep(0.5)

        for i in range(self.args.wait_frames):
            ret, frame = self.cap.read()

        if self.args.read_eeprom:
            self.camera_xu.open(self.camera_names_real[self.current_index])
            self.camera_xu.read_eeprom("inf_eeprom.dat", "mac_eeprom.dat", 4608)
            self.camera_xu.close()
        self.cap.set_width(16320)
        self.cap.set_height(6144)
        self.cap.reStart()
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        for i in range(self.args.wait_frames):
            ret, frame = self.cap.read()

        self.cap.release()
        self.cap = Camera(self.current_index, cv2.CAP_DSHOW)
        self.cap.open()

        for i in range(self.args.wait_frames):
            ret, _ = self.cap.read()

        raw_name = time.strftime('200MP_%Y-%m-%d_%H_%M_%S.raw')
        raw_path = self.save_dir / raw_name
        np.array(frame).tofile(str(raw_path))

        # you can optionally rename the JPG to reflect the 200MP dimensions:
        png_name = time.strftime(f'200MP_preview_%Y-%m-%d_%H_%M_%S.png')
        png_path = self.save_dir / png_name

        logger.info(f"High-res saved: {raw_path}, preview: {png_path}")

        image = convert_200mp(raw_path)
        jpg_path = raw_path.with_suffix(".jpg")
        logger.debug(f"Saving {jpg_path}.jpg")
        image = gray_world_white_balance(image)
        # sRGB)
        img_blurred = cv2.medianBlur(image, 7)
        cv2.imwrite(str(jpg_path), img_blurred)
        cv2.imwrite(str(png_path), img_blurred)

        # height, width = (12288, 16320)
        # img_raw = frame.reshape((height, width))
        # cv2.imwrite("raw.jpg", img_raw)
        # rgb = conv.convert_200mp(data=img_raw, bayer_pattern=cv2.COLOR_BAYER_GB2RGB_VNG)

        # restore original resolution/fps
        self.cap.set_width(orig_w)
        self.cap.set_height(orig_h)
        self.cap.set_fps(orig_fps)
        self.cap.reStart()

        self.update_frame()

        messagebox.showinfo("200 MP Capture",
                            f"Raw saved to:\n{raw_path}\nPreview saved to:\n{jpg_path}")

    def restore_preview_settings(self):
        """ Restores camera to its original preview settings. """
        logger.debug("Restoring preview settings...")
        self.cap.set_width(self.args.width)
        self.cap.set_height(self.args.height)
        self.cap.set_fps(self.args.FrameRate)
        self.cap.reStart()
        # The update_frame loop will continue automatically
        logger.debug("Preview restored.")

    def quit(self):
        if self.cap:
            self.cap.release()
        self.master.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arducam 200MP Camera Controller")
    parser.add_argument('-W', '--width', type=int, default=1280, help="Preview window width.")
    parser.add_argument('-H', '--height', type=int, default=720, help="Preview window height.")
    parser.add_argument('--FrameRate', type=int, default=30, help="Preview frame rate.")
    parser.add_argument('--Focus', type=int, help='Set initial manual focus value (0-4095).')
    parser.add_argument('--Exposure', type=float, help='Set initial manual exposure value (-14 to -1).')
    # parser.add_argument('--VideoCaptureAPI', type=int, default=2, choices=range(len(VIDEO_CAPTURE_APIS)),
    #                     help="OpenCV VideoCapture API. 0:ANY, 1:MSMF, 2:DSHOW, 3:V4L2.")
    parser.add_argument('--reStartTimes', type=int, default=5, help="Number of times to try restarting a failed stream.")
    parser.add_argument('--wait-frames', type=int, default=3, help="Frames to discard after mode change before capturing.")
    parser.add_argument('--read-eeprom', action='store_true', help="Read EEPROM data before 200MP capture.")
    args = parser.parse_args()

    root = tk.Tk()
    app = CameraApp(root, args)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()