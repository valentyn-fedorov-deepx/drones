# client.py
import tkinter as tk
from tkinter import messagebox
import asyncio
import websockets
from PIL import Image, ImageTk
from typing import Union
import numpy as np

from src.project_managers.outputs.project_c_outputs import ShotEvent


class WebSocketClient:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("WebSocket Shot Client")
        self.window.geometry("600x800")
        
        # IP Address input
        self.ip_label = tk.Label(self.window, text="Server IP:")
        self.ip_label.pack(pady=5)
        
        self.ip_entry = tk.Entry(self.window)
        self.ip_entry.insert(0, "localhost")
        self.ip_entry.pack(pady=5)
        
        # Send button
        self.send_button = tk.Button(self.window, text="Send shot message", command=self.send_shot)
        self.send_button.pack(pady=20)
        
        # Response output
        self.response_label = tk.Label(self.window, text="Response:", font=('Arial', 14, 'bold'))
        self.response_label.pack(pady=5)

        self.response_text = tk.Text(self.window, height=10, width=40, font=('Arial', 12))
        self.response_text.pack(pady=5)
        
        # Image display
        self.image_label = tk.Label(self.window)
        self.image_label.pack(pady=10)
        
        # Store the PhotoImage reference
        self.photo = None

    def send_shot(self):
        async def async_send():
            try:
                ip_address = self.ip_entry.get()
                uri = f"ws://{ip_address}:8765"
                
                # with open("/Users/svatoslavdarmograj/Downloads/shot_event.json") as f:
                #     import json
                #     shot_event_data = json.load(f)

                async with websockets.connect(uri) as websocket:
                    await websocket.send("shot")
                    shotevent_data_bytes = await websocket.recv()
                    shot_event = ShotEvent.decode_from_bytes(shotevent_data_bytes)
                    shot_event_txt_message = "\n".join([f"{key}: {value}" for key, value in shot_event.to_dict().items()])
                    
                    
                    # print(shot_event_data)
                    # shot_event_str = '\n'.join([f"{key}: {value}" for key, value in shot_event_data.items()])
                    # self.update_response(f"Received image: {image_filename}")
                    # self.update_response(shot_event_str)
                    self.update_response(shot_event_txt_message)

                    self.load_and_display_image(shot_event.jpeg_bit_stream)
                    
            except Exception as e:
                self.show_error(str(e))

        asyncio.run(async_send())

    def update_response(self, response):
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, response)

    def load_and_display_image(self, image: Union[str, Image.Image, np.ndarray]):
        try:
            # Load and resize image
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # Resize while maintaining aspect ratio
            display_size = (500, 500)  # Maximum display size
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and store reference
            self.photo = ImageTk.PhotoImage(image)
            
            # Update image in label
            self.image_label.configure(image=self.photo)
            
        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")

    def show_error(self, error_message):
        messagebox.showerror("Error", f"Error: {error_message}")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    client = WebSocketClient()
    client.run()