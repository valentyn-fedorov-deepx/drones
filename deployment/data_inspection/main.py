import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

import gradio as gr
import numpy as np
import cv2

from configs.data_pixel_tensor import ELEMENTS_NAMES_WITH_RAW, DATA_PIXEL_TENSOR_BACKEND
from src.offline_utils.frame_source import FrameSource


class VideoFrameViewer:
    """
    A Gradio-based video frame viewer for browsing frame sequences
    with a slider and Previous/Next buttons, plus playback mode.
    """

    def __init__(self, main_folder_path: str):
        self.main_folder_path = Path(main_folder_path)
        self.current_frame_source = None
        self.current_subdirectory = None
        self.current_dir_info = {}
        self.element_name = "view_img"
        # Downscale percentage (0–90)
        self.downscale_percent = 0

        self.playback_active = False
        self.playback_fps = 10  # default fps

        self.app = self._create_interface()

    def _get_subdirectories(self) -> List[str]:
        if not self.main_folder_path.exists():
            return []
        return sorted([item.name for item in self.main_folder_path.iterdir() if item.is_dir()])

    def _get_directory_info(self, subdir_path: Path) -> Dict[str, Any]:
        if not subdir_path.exists():
            return {"size": 0}
        total_size = sum(f.stat().st_size for f in subdir_path.rglob('*') if f.is_file())
        return {"size": total_size}

    def _format_file_size(self, size_bytes: int) -> str:
        if size_bytes < 1024.0: return f"{size_bytes} B"
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return "0 B"
    
    def _playback_button_states(self):
        """Return (play_btn_update, pause_btn_update) reflecting current state."""
        if self.playback_active:
            # Playing: Play disabled (neutral), Pause enabled (red)
            return (
                gr.update(interactive=False, variant="secondary"),
                gr.update(interactive=True,  variant="stop"),
            )
        else:
            # Paused: Play enabled (primary), Pause disabled (neutral)
            return (
                gr.update(interactive=True,  variant="primary"),
                gr.update(interactive=False, variant="secondary"),
            )

    def _build_info_text(self, frame_number: int, frame_shape) -> str:
        if not self.current_subdirectory or not self.current_dir_info:
            return "No subdirectory selected."

        playback = "On" if self.playback_active else "Off"
        fps = self.playback_fps

        return f"""
**Directory:** {self.current_subdirectory}\n
**Total Frames:** {self.current_dir_info['total_frames']}\n
**Directory Size:** {self.current_dir_info['size']}\n
**Current Frame:** {frame_number} / {self.current_dir_info['total_frames']}\n
**Image Dimensions (displayed):** {frame_shape[1]} x {frame_shape[0]}\n
**Decrease size:** {self.downscale_percent}%\n
**Playback:** {playback}\n
**FPS:** {fps}
        """.strip()

    def _apply_downscale(self, frame: np.ndarray) -> np.ndarray:
        p = int(self.downscale_percent)
        if p <= 0:
            return frame

        h, w = frame.shape[:2]
        scale = (100 - p) / 100.0
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Ensure display-friendly dtype
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame8 = np.clip(frame, 0.0, 1.0)
                frame8 = (frame8 * 255.0).round().astype(np.uint8)
            else:
                frame8 = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame8 = frame

        # Use INTER_AREA for downscaling (best quality)
        resized = cv2.resize(frame8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    def rescan_folders(self) -> gr.update:
        subdirs = self._get_subdirectories()
        choices = subdirs if subdirs else ["No subdirectories found"]
        value = subdirs[0] if subdirs else None
        return gr.update(choices=choices, value=value)

    def select_subdirectory(self, selected_subdir: str) -> Tuple[gr.update, gr.update, str]:
        if not selected_subdir or selected_subdir == "No subdirectories found":
            self.current_dir_info = {}
            return gr.update(value=None), gr.update(interactive=False, value=1), "No subdirectory selected"

        try:
            subdir_path = self.main_folder_path / selected_subdir
            self.current_frame_source = FrameSource(str(subdir_path))
            self.current_subdirectory = selected_subdir

            total_frames = len(self.current_frame_source)
            if total_frames == 0:
                return gr.update(value=None), gr.update(interactive=False), "No frames found in directory"

            dir_info = self._get_directory_info(subdir_path)
            self.current_dir_info = {
                "size": self._format_file_size(dir_info['size']),
                "total_frames": total_frames
            }

            data_tensor = self.current_frame_source[0]
            frame = data_tensor[self.element_name]
            if DATA_PIXEL_TENSOR_BACKEND == 'torch':
                frame = frame.cpu().numpy()

            frame = frame / data_tensor.max_value
            frame = np.uint8(frame * 255)

            disp = self._apply_downscale(frame)
            info_text = self._build_info_text(frame_number=1, frame_shape=disp.shape)

            return (
                gr.update(value=disp),
                gr.update(maximum=total_frames, value=1, interactive=True),
                info_text
            )
        except Exception as e:
            self.current_dir_info = {}
            return gr.update(value=None), gr.update(interactive=False), f"Error loading subdirectory: {str(e)}"

    def delete_subdirectory(self, selected_subdir: str) -> Tuple[gr.update, gr.update, gr.update, str]:
        if not selected_subdir or selected_subdir == "No subdirectories found":
            return gr.update(), gr.update(), gr.update(), "No subdirectory selected for deletion"

        try:
            shutil.rmtree(self.main_folder_path / selected_subdir)
            self.current_frame_source = None
            self.current_subdirectory = None
            self.current_dir_info = {}

            subdirs = self._get_subdirectories()
            choices = subdirs if subdirs else ["No subdirectories found"]
            value = subdirs[0] if subdirs else None

            return (
                gr.update(choices=choices, value=value),
                gr.update(value=None),
                gr.update(interactive=False, value=1),
                f"Directory '{selected_subdir}' deleted successfully"
            )
        except Exception as e:
            return gr.update(), gr.update(), gr.update(), f"Error deleting directory: {str(e)}"

    def update_frame(self, frame_number: int) -> Tuple[gr.update, str]:
        if not self.current_frame_source:
            return gr.update(value=None), "No subdirectory selected"
        try:
            frame_idx = int(frame_number) - 1
            if not (0 <= frame_idx < len(self.current_frame_source)):
                return gr.update(), "Frame index out of range"

            data_tensor = self.current_frame_source[frame_idx]
            frame = data_tensor[self.element_name]
            if DATA_PIXEL_TENSOR_BACKEND == 'torch':
                frame = frame.cpu().numpy()
                
            frame = frame / data_tensor.max_value
            frame = np.uint8(frame * 255)
            disp = self._apply_downscale(frame)
            info_text = self._build_info_text(frame_number=frame_number, frame_shape=disp.shape)

            return gr.update(value=disp), info_text
        except Exception as e:
            return gr.update(), f"Error loading frame: {str(e)}"

    def next_frame(self, current_frame_number: int) -> gr.update:
        if not self.current_frame_source:
            return gr.update()
        total_frames = len(self.current_frame_source)
        new_frame_number = min(int(current_frame_number) + 1, total_frames)
        return gr.update(value=new_frame_number)

    def previous_frame(self, current_frame_number: int) -> gr.update:
        if not self.current_frame_source:
            return gr.update()
        new_frame_number = max(int(current_frame_number) - 1, 1)
        return gr.update(value=new_frame_number)

    def select_element(self, selected_element: str, frame_number: int) -> Tuple[gr.update, str]:
        self.element_name = selected_element
        if not self.current_frame_source:
            return gr.update(), f"Element set to '{selected_element}'. Select a subdirectory."
        return self.update_frame(frame_number)

    # Change downscale percent and refresh the current frame
    def change_downscale(self, percent: int, frame_number: int) -> Tuple[gr.update, str]:
        self.downscale_percent = int(percent)
        return self.update_frame(frame_number)

    def play(self, fps: int) -> Tuple[gr.update, gr.update, gr.update]:
        """Start playback: activate timer and set interval to 1/fps."""
        # If nothing selected, don't start
        if not self.current_frame_source:
            self.playback_active = False
            return gr.update(active=False), *self._playback_button_states()

        self.playback_active = True
        self.playback_fps = max(1, int(fps))
        period = max(0.01, 1.0 / self.playback_fps)
        return gr.update(active=True, value=period), *self._playback_button_states()

    def pause(self) -> Tuple[gr.update, gr.update, gr.update]:
        """Pause playback (deactivate timer)."""
        self.playback_active = False
        return gr.update(active=False), *self._playback_button_states()

    def change_fps(self, fps: int) -> Tuple[gr.update, gr.update, gr.update]:
        """Update playback fps & timer period (if running)."""
        self.playback_fps = max(1, int(fps))
        period = max(0.01, 1.0 / self.playback_fps)
        # Buttons don't change state here, but we still return them for consistency
        return gr.update(value=period), *self._playback_button_states()

    def playback_step(
        self, current_frame_number: int, loop: bool
    ) -> Tuple[gr.update, str, gr.update, gr.update, gr.update, gr.update]:
        if not self.current_frame_source:
            self.playback_active = False
            return (
                gr.update(), "No subdirectory selected", gr.update(), gr.update(active=False),
                *self._playback_button_states()
            )

        total_frames = len(self.current_frame_source)
        cur = int(current_frame_number)
        nxt = cur + 1

        if nxt > total_frames:
            if loop:
                nxt = 1
                img_u, info = self.update_frame(nxt)
                return img_u, info, gr.update(value=nxt), gr.update(), *self._playback_button_states()
            else:
                img_u, info = self.update_frame(total_frames)
                self.playback_active = False
                return img_u, info, gr.update(value=total_frames), gr.update(active=False), *self._playback_button_states()

        img_u, info = self.update_frame(nxt)
        return img_u, info, gr.update(value=nxt), gr.update(), *self._playback_button_states()
    # ------------------------------------------------------------------------

    def _create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Video Frame Viewer", theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# Video Frame Viewer\n**Main Folder:** `{self.main_folder_path}`")
            with gr.Row():
                with gr.Column(scale=1):
                    # Controls and info sections
                    gr.Markdown("## Controls")
                    rescan_btn = gr.Button("🔄 Rescan Folders", variant="secondary")
                    subdirectory_dropdown = gr.Dropdown(
                        label="Select Subdirectory",
                        choices=self._get_subdirectories() or ["No subdirectories found"],
                        interactive=True,
                        value=None
                    )
                    element_dropdown = gr.Dropdown(
                        label="Select Element",
                        choices=ELEMENTS_NAMES_WITH_RAW or ["No elements found"],
                        value=self.element_name,
                        interactive=True
                    )
                    # Decrease size slider
                    downscale_slider = gr.Slider(
                        label="Decrease size (%)",
                        minimum=0,
                        maximum=90,
                        value=0,
                        step=1,
                        interactive=True
                    )
                    delete_btn = gr.Button("🗑️ Delete Current Subdirectory", variant="stop")

                    gr.Markdown("## Navigation")
                    frame_slider = gr.Slider(
                        label="Frame", minimum=1, maximum=100, value=1, step=1, interactive=False
                    )
                    with gr.Row():
                        prev_frame_btn = gr.Button("◀️ Previous", variant="secondary")
                        next_frame_btn = gr.Button("Next ▶️", variant="secondary")

                    # --- NEW: Playback controls ---
                    gr.Markdown("## Playback")
                    with gr.Row():
                        play_btn  = gr.Button("▶️ Play",  variant="primary",  interactive=True)
                        pause_btn = gr.Button("⏸️ Pause", variant="secondary", interactive=False)

                    fps_slider = gr.Slider(
                        label="Playback FPS", minimum=1, maximum=60, value=self.playback_fps, step=1, interactive=True
                    )
                    loop_checkbox = gr.Checkbox(label="Loop", value=True)
                    timer = gr.Timer(value=max(0.01, 1.0 / self.playback_fps), active=False)

                    gr.Markdown("## Information")
                    info_text = gr.Markdown("No subdirectory selected")

                with gr.Column(scale=2):
                    gr.Markdown("## Frame Viewer")
                    image_display = gr.Image(
                        label="Current Frame",
                        type="numpy",
                        interactive=False,
                        height=600
                    )

            # Event handlers
            rescan_btn.click(fn=self.rescan_folders, outputs=subdirectory_dropdown).then(
                fn=self.select_subdirectory,
                inputs=subdirectory_dropdown,
                outputs=[image_display, frame_slider, info_text],
                show_progress="hidden",
                )

            subdirectory_dropdown.change(
                fn=self.select_subdirectory,
                inputs=subdirectory_dropdown,
                outputs=[image_display, frame_slider, info_text],
                show_progress="hidden",
            )

            element_dropdown.change(
                fn=self.select_element,
                inputs=[element_dropdown, frame_slider],
                outputs=[image_display, info_text],
                show_progress="hidden",
            )

            downscale_slider.change(
                fn=self.change_downscale,
                inputs=[downscale_slider, frame_slider],
                outputs=[image_display, info_text],
                show_progress="hidden",
            )

            delete_btn.click(
                fn=self.delete_subdirectory,
                inputs=subdirectory_dropdown,
                outputs=[subdirectory_dropdown, image_display, frame_slider, info_text],
                show_progress="hidden",
            ).then(
                fn=self.select_subdirectory,
                inputs=subdirectory_dropdown,
                outputs=[image_display, frame_slider, info_text],
                show_progress="hidden",
                )

            frame_slider.change(
                fn=self.update_frame,
                inputs=frame_slider,
                outputs=[image_display, info_text],
                show_progress="hidden",
            )

            prev_frame_btn.click(fn=self.previous_frame, inputs=frame_slider, outputs=frame_slider)
            next_frame_btn.click(fn=self.next_frame, inputs=frame_slider, outputs=frame_slider)
            
            play_btn.click(
                fn=self.play, inputs=fps_slider, outputs=[timer, play_btn, pause_btn]
            )
            pause_btn.click(
                fn=self.pause, outputs=[timer, play_btn, pause_btn]
            )
            fps_slider.change(
                fn=self.change_fps, inputs=fps_slider, outputs=[timer, play_btn, pause_btn]
            )

            timer.tick(
                fn=self.playback_step,
                inputs=[frame_slider, loop_checkbox],
                outputs=[image_display, info_text, frame_slider, timer, play_btn, pause_btn],
                show_progress="hidden",
            )

        return app

    def launch(self, **kwargs):
        self.app.launch(**kwargs)


# Main execution block remains the same
if __name__ == "__main__":
    DEFAULT_MAIN_FOLDER_PATH = "/sdb-disk/vyzai/data/pxi_source/2025.07.25_Second_Day_ProjectAR"
    parser = argparse.ArgumentParser(description="Data Viewer")
    parser.add_argument(
        "--main-folder", "-m", dest="main_folder_path", type=str,
        default=DEFAULT_MAIN_FOLDER_PATH,
        help=f"Path to the main folder (default: {DEFAULT_MAIN_FOLDER_PATH})"
    )
    args = parser.parse_args()
    viewer = VideoFrameViewer(args.main_folder_path)
    viewer.launch(share=False, debug=True, server_name="0.0.0.0")
