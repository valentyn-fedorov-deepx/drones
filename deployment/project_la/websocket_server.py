import asyncio
import json
import websockets
import websockets.asyncio
import websockets.asyncio.server
from typing import Dict, Set, Optional
from loguru import logger
import threading

import queue
from argparse import ArgumentParser

from src.data_pixel_tensor import DataPixelTensor
from src.project_managers.project_la_manager import ProjectLAManager
from deployment.project_la.constants import GET_ACTION, CONTINUOUS_ACTION, STOP_CONTINUOUS_ACTION, SETUP_LOCATION_ACTION, SETUP_CAMERA_ACTION
from deployment.project_la.request_processor import prepare_data_response
from deployment.project_la.websocket_test import DUMMY_DETECTED_OBJECTS, DUMMY_DATA_TENSOR
from deployment.project_la.request_models import validate_request_with_errors, StopContinuousRequestData, GetRequestData, ContinuousRequestData, SetupCameraRequestData, SetupLocationRequestData
from deployment.networking_utils import get_local_ip
from deployment.project_la.constants import DATA_RESPONSE_CODE, STATUS_RESPONSE_CODE

# Configure loguru
logger.add(
    "logs/ProjectLA_websocket_server.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
)


def parse_args():
    parser = ArgumentParser("Script that will start the project LA server")

    parser.add_argument("--data-source", default="live", help="For Live camera. It should be 'live'. To run from video/pxis you will need to specify path to them.")
    parser.add_argument("--port", default="8765", help="Port on which the server will be running")

    return parser.parse_args()


class DummyProcessor:
    def __init__(self):
        self.latest_data_tensor = DUMMY_DATA_TENSOR
        self.detected_objects = DUMMY_DETECTED_OBJECTS

    def process(self, input_data: DataPixelTensor):
        pass


def inference_worker(result_queue, camera_params_queue, location_queue,
                     config_path, device, data_source_name="live"):
    logger.info("Started inference worker")
    if data_source_name == 'live':
        from src.camera import LiveCamera
        data_source = LiveCamera(exposure_mode="continuous", depth=8)
    else:
        from src.offline_utils.frame_source import FrameSource
        data_source = FrameSource(data_source_name, 0, None, loop=True)

    processor = ProjectLAManager(config_path=config_path,
                                 device=device)

    while True:
        try:
            data_tensor = next(data_source)
            processor.process(data_tensor)

            results = (processor.latest_data_tensor, processor.detected_objects)

            # Non-blocking put: if queue is full, discard oldest item first
            if result_queue.full():
                try:
                    result_queue.get_nowait()  # Discard oldest
                    logger.warning("Result queue was full, discarded oldest item.")
                except queue.Empty:
                    pass  # Should not happen if full() is true, but good practice
            result_queue.put(results)

            if not location_queue.empty():
                try:
                    location_data = location_queue.get_nowait()
                    processor.longitude = location_data.longitude
                    processor.latitude = location_data.latitude
                    processor.altitude = location_data.altitude
                    processor.heading = location_data.heading
                    logger.info(f"Inference worker updated location: {location_data}")
                except queue.Empty:
                    pass  # Another process might have consumed it

            if not camera_params_queue.empty():
                camera_params = camera_params_queue.get_nowait()

                if camera_params.exposure_mode:
                    data_source.set_exposure_mode(camera_params.exposure_mode)

                if camera_params.exposure_value and camera_params.exposure_mode == 'off':
                    data_source.set_exposure(camera_params.exposure_value)

                if camera_params.gain_value:
                    data_source.set_gain(camera_params.gain_value)

                if camera_params.focal_length_mm:
                    processor.change_focal_length(camera_params.focal_length_mm)

                if camera_params.pixel_format:
                    data_source.set_pixel_format(camera_params.pixel_format)

        except StopIteration:
            logger.warning("Data source exhausted.")
            break
        except Exception as e:
            logger.error(f"Error in inference_worker loop: {e}", exc_info=True)
            # Depending on the error, you might want to break or sleep before retrying
            asyncio.sleep(1) # Avoid tight loop on persistent errors


class ProjectLAServer:
    def __init__(self):
        self.continuous_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        self.active_connections: Set[websockets.asyncio.server.ServerConnection] = set()
        self.send_locks: Dict[str, asyncio.Lock] = {}

        self.result_queue = queue.Queue(maxsize=10)
        self.camera_params_queue = queue.Queue(maxsize=10)
        self.location_queue = queue.Queue(maxsize=10)

        logger.info("ProjectLAServer initialized")
        self.latest_data_tensor = None
        self.detected_objects = list()
        self._data_updater_task_handle: Optional[asyncio.Task] = None

    async def _data_updater_task(self):
        """
        Dedicated asyncio task to fetch data from the multiprocessing queue
        and update server attributes.
        """
        logger.info("Data updater task started.")
        while True:
            try:
                # Run the blocking get() in a separate thread
                # Pass the instance method directly or use a lambda if it captures self correctly.
                # Here, self.result_queue.get is a bound method, so it's fine.
                latest_data, detected_objs = await asyncio.to_thread(self.result_queue.get)

                self.latest_data_tensor = latest_data
                self.detected_objects = detected_objs
                # logger.debug(f"Updated server data: {len(self.detected_objects)} objects.") # Can be very verbose
            except Exception as e: # Catch broader exceptions if queue.get() fails for other reasons
                logger.error(f"Error in data updater task: {e}", exc_info=True)
                # If the queue is closed (e.g., worker process died), this task might exit or log continuously.
                # Consider a short sleep to prevent tight loop logging on persistent errors.
                await asyncio.sleep(0.1)

    async def start_updater_task(self):
        """Starts the data updater task."""
        if self._data_updater_task_handle is None or self._data_updater_task_handle.done():
            self._data_updater_task_handle = asyncio.create_task(self._data_updater_task())
            logger.info("Data updater task scheduled to run.")

    async def stop_updater_task(self):
        """Stops the data updater task if running."""
        if self._data_updater_task_handle and not self._data_updater_task_handle.done():
            self._data_updater_task_handle.cancel()
            try:
                await self._data_updater_task_handle
            except asyncio.CancelledError:
                logger.info("Data updater task cancelled successfully.")
            except Exception as e:
                logger.error(f"Exception during data updater task cancellation: {e}")
            self._data_updater_task_handle = None

    async def register(self, websocket: websockets.asyncio.server.ServerConnection):
        self.active_connections.add(websocket)
        # Initialize the continuous task dict for this connection
        websocket_id = str(id(websocket))
        if websocket_id not in self.continuous_tasks:
            self.continuous_tasks[websocket_id] = {}

        self.send_locks[websocket_id] = asyncio.Lock()
        logger.info(f"New connection registered. Total active connections: {len(self.active_connections)}")
        message = dict(status="success",
                       data="Successfully established connection",
                       response_code=STATUS_RESPONSE_CODE)
        await self.safe_send(websocket, json.dumps(message))

    async def unregister(self, websocket: websockets.asyncio.server.ServerConnection):
        websocket_id = str(id(websocket))
        self.active_connections.remove(websocket)
        # Cancel and remove any continuous tasks for this connection
        if websocket_id in self.continuous_tasks:
            for task_id, task in self.continuous_tasks[websocket_id].items():
                task.cancel()
                logger.debug(f"Cancelled continuous task {task_id} for websocket {websocket_id}")
            del self.continuous_tasks[websocket_id]
        logger.info(f"Connection unregistered. Total active connections: {len(self.active_connections)}")

    async def safe_send(self, websocket: websockets.asyncio.server.ServerConnection, message: str):
        websocket_id = str(id(websocket))
        lock = self.send_locks[websocket_id]
        async with lock:
            await websocket.send(message)

    async def continuous_data_sender(self, websocket: websockets.asyncio.server.ServerConnection,
                                     continuous_request_data: ContinuousRequestData):
        interval = continuous_request_data.repeat_time_seconds
        task_id = continuous_request_data.task_id # For logging
        websocket_id = str(id(websocket)) # For logging
        logger.info(f"Task {task_id} for {websocket_id}: Continuous data sender started with interval {interval}s.")
        try:
            while True:
                # No longer calls get_data_from_manager. Uses server's attributes directly.
                # These attributes are updated by _data_updater_task.
                if self.latest_data_tensor is None and not self.detected_objects:
                    logger.debug(f"Task {task_id} for {websocket_id}: No data available yet, sleeping briefly.")
                    await asyncio.sleep(0.1) # Wait a bit if no data has ever arrived
                    continue

                response_data = prepare_data_response(self.detected_objects, # Already up-to-date
                                                      self.latest_data_tensor, # Already up-to-date
                                                      continuous_request_data.response_items)

                full_response = dict(status="success",
                                     data=response_data,
                                     response_code=DATA_RESPONSE_CODE)
                # logger.warning(f"Task {task_id} for {websocket_id}: Sending continuous data update.") # warning level too high
                logger.debug(f"Task {task_id} for {websocket_id}: Sending continuous data update with {len(self.detected_objects)} objects.")
                
                await self.safe_send(websocket, json.dumps(full_response))
                
                if interval is not None and interval > 0:
                    await asyncio.sleep(interval)
                else: # If interval is None, 0 or negative, send once and stop, or send as fast as possible (yield)
                    # If the intent is to send as fast as possible, yielding is important
                    await asyncio.sleep(0) # Yield control to event loop
                    if interval is not None:
                        pass
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} for {websocket_id}: Continuous data sender task cancelled.")
            raise # Re-raise to ensure task is properly cleaned up
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Task {task_id} for {websocket_id}: Connection closed during continuous send.")
        except Exception as e:
            logger.error(f"Task {task_id} for {websocket_id}: Error in continuous_data_sender: {e}", exc_info=True)
            # Depending on error, might want to stop this task or send an error to client.


    async def process_set_continuous_request(self, websocket, data: ContinuousRequestData):
        task_id = data.task_id
        websocket_id = str(id(websocket))

        task = asyncio.create_task(
            self.continuous_data_sender(websocket, data)
        )

        # Save the task in the dict for this websocket
        if websocket_id not in self.continuous_tasks:
            self.continuous_tasks[websocket_id] = {}
        self.continuous_tasks[websocket_id][task_id] = task
        logger.info(f"Started new continuous data task {task_id} with interval sec for websocket {websocket_id}")

        # Return the task_id to the client
        response = dict(status="success",
                        data=f"Created task with task_id: {task_id}",
                        response_code=STATUS_RESPONSE_CODE)
        # await websocket.send(json.dumps(response))
        await self.safe_send(websocket, json.dumps(response))

    async def process_get_data_request(self, websocket: websockets.asyncio.server.ServerConnection,
                                       data: GetRequestData):
        response_items = data.response_items

        response_data = prepare_data_response(self.detected_objects,
                                              self.latest_data_tensor,
                                              response_items)

        logger.debug(f"Sending single data update: {response_data}")
        full_response = dict(data=response_data,
                             response_code=DATA_RESPONSE_CODE)

        await self.safe_send(websocket, json.dumps(full_response))

    async def process_stop_continuous_request(self, websocket: websockets.asyncio.server.ServerConnection,
                                              data: StopContinuousRequestData):
        websocket_id = str(id(websocket))
        task_id = data.task_id

        if websocket_id in self.continuous_tasks and task_id in self.continuous_tasks[websocket_id]:
            self.continuous_tasks[websocket_id][task_id].cancel()
            del self.continuous_tasks[websocket_id][task_id]
            logger.info(f"Stopped continuous data task {task_id} for websocket {websocket_id}")
            response = dict(status="success",
                            data=f"Stopped task_id: {task_id}",
                            response_code=STATUS_RESPONSE_CODE)
        else:
            logger.warning(f"Task id {task_id} not found for websocket {websocket_id}")
            response = dict(status="error",
                            data="task_id not found",
                            response_code=STATUS_RESPONSE_CODE)
        # await websocket.send(json.dumps(response))
        await self.safe_send(websocket, json.dumps(response))

    async def process_location_request(self, websocket: websockets.asyncio.server.ServerConnection,
                                       data: SetupLocationRequestData):
        try:
            # Non-blocking put, discard if full (or handle differently, e.g., error back)
            if self.location_queue.full():
                 logger.warning("Location queue is full. Discarding new location data.")
                 # Or: self.location_queue.get_nowait() to make space, then put.
                 # Or: respond with an error.
                 response = dict(status="error", data="Location queue full, try again later.", response_code=STATUS_RESPONSE_CODE)
                 await self.safe_send(websocket, json.dumps(response))
                 return

            self.location_queue.put_nowait(data) # Use put_nowait for async context, handle Full exception
            logger.info(f"Location data {data} queued for processing.")
            response = dict(status="success",
                            data="Location update queued.",
                            response_code=STATUS_RESPONSE_CODE)
        except queue.Full: # Should be caught by the check above, but defensive.
            logger.error("Location queue is full (put_nowait). This should have been caught.")
            response = dict(status="error", data="Failed to queue location update (queue full).", response_code=STATUS_RESPONSE_CODE)
        except Exception as err:
            logger.error(f"Error processing location request: {err}", exc_info=True)
            response = dict(status="error",
                            data=str(err),
                            response_code=STATUS_RESPONSE_CODE)

        await self.safe_send(websocket, json.dumps(response))


    async def process_camera_request(self, websocket: websockets.asyncio.server.ServerConnection,
                                     data: SetupCameraRequestData):
        try:
            if self.camera_params_queue.full():
                logger.warning("Camera params queue is full. Discarding new camera data.")
                response = dict(status="error", data="Camera params queue full, try again later.", response_code=STATUS_RESPONSE_CODE)
                await self.safe_send(websocket, json.dumps(response))
                return

            self.camera_params_queue.put_nowait(data)
            logger.info(f"Camera parameters {data} queued for processing.")
            response = dict(status="success",
                            data="Camera parameters update queued.",
                            response_code=STATUS_RESPONSE_CODE)
        except queue.Full:
            logger.error("Camera params queue is full (put_nowait).")
            response = dict(status="error", data="Failed to queue camera parameters (queue full).", response_code=STATUS_RESPONSE_CODE)
        except Exception as err:
            logger.error(f"Error processing camera request: {err}", exc_info=True)
            response = dict(status="error",
                            data=f"Couldn't change the camera parameters. {err}",
                            response_code=STATUS_RESPONSE_CODE)
        await self.safe_send(websocket, json.dumps(response))


    async def execute_request(self, websocket: websockets.asyncio.server.ServerConnection,
                              request):
        if request.action == GET_ACTION:
            await self.process_get_data_request(websocket,
                                                request.request_data)
        elif request.action == CONTINUOUS_ACTION:
            await self.process_set_continuous_request(websocket,
                                                      request.request_data)
        elif request.action == STOP_CONTINUOUS_ACTION:
            await self.process_stop_continuous_request(websocket,
                                                       request.request_data)
        elif request.action == SETUP_LOCATION_ACTION:
            await self.process_location_request(websocket,
                                                request.request_data)
        elif request.action == SETUP_CAMERA_ACTION:
            await self.process_camera_request(websocket,
                                              request.request_data)

    async def handle_message(self, websocket: websockets.asyncio.server.ServerConnection,
                             message: str):
        try:
            data = json.loads(message)
            is_valid, request = validate_request_with_errors(data)
            if not is_valid:
                error_msg = dict(status="error",
                                 data=str(request),
                                 response_code=STATUS_RESPONSE_CODE)
                logger.error(f"Failed to decode JSON message: {message}")
                # await websocket.send(json.dumps(error_msg))
                await self.safe_send(websocket, json.dumps(error_msg))
            else:
                await self.execute_request(websocket, request)

        except json.JSONDecodeError:
            error_msg = dict(status="error",
                             data="Invalid JSON",
                             response_code=STATUS_RESPONSE_CODE)
            logger.error(f"Failed to decode JSON message: {message}")
            await self.safe_send(websocket, json.dumps(error_msg))

    async def handle_connection(self, websocket: websockets.asyncio.server.ServerConnection):
        await self.register(websocket)

        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            await self.unregister(websocket)


async def main(port_str: str, data_source_name: str): # Added type hints
    port = int(port_str) # Convert port to int
    server = ProjectLAServer()

    config_path: str = "configs/project_la/manager.yaml" # Ensure this path is correct
    device: str = "cpu" # or "cuda" / "mps" etc.

    # Start the inference worker process
    # Ensure Process is created only once if main can be re-entered or server restarted.
    # For this script structure, it's fine.
    # logger.info()
    inference_proc = threading.Thread(target=inference_worker, args=(server.result_queue,
                                                                server.camera_params_queue,
                                                                server.location_queue,
                                                                config_path, device, data_source_name),
                                daemon=True) # Set as daemon so it exits when main process exits
    inference_proc.start()
    # logger.info(f"Inference worker process started (PID: {inference_proc.pid}).")

    # Start the server's internal data updater task
    await server.start_updater_task()

    try:
        async with websockets.asyncio.server.serve(server.handle_connection, "0.0.0.0", port):
            logger.info(f"WebSocket server started at ws://{get_local_ip()}:{port}")
            await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Server shutting down due to KeyboardInterrupt...")
    except Exception as e:
        logger.critical(f"Server failed to start or encountered critical error: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up server resources...")
        await server.stop_updater_task() # Stop the updater task

        # Clean up active connections and their tasks
        # Create a list of connections to avoid issues if unregister modifies the set
        active_connections_copy = list(server.active_connections)
        for ws in active_connections_copy:
            await server.unregister(ws) # This will cancel associated tasks

        if inference_proc and inference_proc.is_alive():
            logger.info("Terminating inference worker process...")
            # inference_proc.terminate() # Send SIGTERM
            inference_proc.join(timeout=5) # Wait for it to exit
            # if inference_proc.is_alive():
                # logger.warning("Inference worker did not terminate gracefully, killing.")
                # inference_proc.kill() # Send SIGKILL if still alive
            logger.info("Inference worker process stopped.")
        logger.info("Server shutdown complete.")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.port, args.data_source))
