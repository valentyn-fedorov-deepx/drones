import asyncio
import websockets
import logging
import sys

from src.project_managers.project_c_manager import ProjectCManager
from src.offline_utils.frame_source import FrameSource
from src.camera.camera import LiveCamera


SERVER_PORT = 8765
SERVER_HOST = '0.0.0.0'


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('websockets')
logger.setLevel(logging.DEBUG)


class ProjectCServer:
    def __init__(self, source: str = 'live'):
        self._project_c_manager = ProjectCManager("configs/", "models/",
                                                  device='cpu')
        self._live_source = source == 'live'
        self._latest_packet_id = 0
        if self._live_source:
            self._frame_source = LiveCamera()
        else:
            self._frame_source = FrameSource(source, None, None)
        self._running = False

    async def handle_connection(self, websocket):
        try:
            async for message in websocket:  # Changed to handle multiple messages
                if message == "shot":
                    shot_message = self._project_c_manager.process_shot()
                    shot_message.packet_id = self._latest_packet_id
                    self._latest_packet_id += 1
                    shot_message_encoded = shot_message.encode_to_bytes()
                    await websocket.send(shot_message_encoded)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error in handle_connection: {str(e)}")

    async def get_data(self):
        logger.info("Starting processing data")
        try:
            while self._running:
                item = next(self._frame_source)
                await asyncio.sleep(0.01)
                self._project_c_manager.process(item)
                await asyncio.sleep(0.01)
                logger.debug("Processed frame")
        except Exception as e:
            logger.error(f"Error in get_data: {str(e)}")

    async def start_server(self):
        try:
            self._running = True

            # Start data gathering task
            data_task = asyncio.create_task(self.get_data())

            # Start WebSocket server
            async with websockets.serve(
                self.handle_connection, 
                SERVER_HOST, 
                SERVER_PORT,
                ping_interval=None,
                compression=None,  # Disable compression for testing
                max_size=2**25,   # 1MB max message size
                close_timeout=25,  # 10 seconds timeout
            ) as server:
                logger.info(f"WebSocket server started on ws://{SERVER_HOST}:{SERVER_PORT}")
                await asyncio.Future()  # run forever
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            self._running = False
            await asyncio.gather(data_task, return_exceptions=True)


if __name__ == "__main__":
    try:
        test_offline_source = '/home/jetson/project_c/test_data/100m_50mm/100m.mp4'
        test_offline_source = "live"
        server = ProjectCServer(source=test_offline_source)
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
