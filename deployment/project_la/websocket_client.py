import asyncio
import json
import websockets
import websockets.asyncio
import websockets.asyncio.client
from typing import Optional

from deployment.project_la.constants import GET_ACTION, CONTINUOUS_ACTION, STOP_CONTINUOUS_ACTION, DATA_RESPONSE_CODE, STATUS_RESPONSE_CODE


class DataClient:
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket: Optional[websockets.asyncio.client.ClientConnection] = None
        self.running_continuous: bool = False
        self.data_responses_received = 0

    async def connect(self):
        self.websocket = await websockets.connect(self.uri)
        print(f"Connected to {self.uri}")

    async def get_people(self):
        if not self.websocket:
            raise ConnectionError("Not connected to server")

        request = {
            "action": GET_ACTION,
            "request_data": {
                "response_items": {
                    "object_names": ["person"],
                    "location": True,
                    "images": {
                        "visualized": False, 
                        "n_z": False,
                        "n_xy": False,
                        "n_xyz": False,
                        "raw": False
                    }
                }
            }
        } 

        await self.websocket.send(json.dumps(request))

    async def get_cars(self):
        if not self.websocket:
            raise ConnectionError("Not connected to server")

        request = {
            "action": GET_ACTION,
            "request_data": {
                "response_items": {
                    "object_names": ["car"],
                    "location": True,
                    "images": {
                        "visualized": False, 
                        "n_z": False,
                        "n_xy": False,
                        "n_xyz": False,
                        "raw": False
                    }
                }
            }
        } 

        await self.websocket.send(json.dumps(request))

    async def get_drones(self):
        if not self.websocket:
            raise ConnectionError("Not connected to server")

        request = {
            "action": GET_ACTION,
            "request_data": {
                "response_items": {
                    "object_names": ["drone"],
                    "location": True,
                    "images": {
                        "visualized_data": False, 
                        "n_z": False,
                        "n_xy": False,
                        "n_xyz": False,
                        "raw": False
                    }
                }
            }
        } 

        await self.websocket.send(json.dumps(request))

    async def start_continuous_data_location(self, task_id: str, interval: int = 1):
        if not self.websocket:
            raise ConnectionError("Not connected to server")

        request = {
            "action": CONTINUOUS_ACTION,
            "request_data": {
                "repeat_time_seconds": interval,
                "task_id": task_id,
                "duration": None,
                "response_items": {
                    "object_names": ["person", "car"],
                    "location": True,
                    "images": {
                        "visualization": False,
                        "n_z": False,
                        "n_xy": False,
                        "n_xyz": False,
                        "raw": False
                    }
                }
            }
        }

        await self.websocket.send(json.dumps(request))

    async def start_continuous_data_image(self, task_id: str, interval: int = 1):
        if not self.websocket:
            raise ConnectionError("Not connected to server")

        request = {
            "action": CONTINUOS_ACTION,
            "request_data": {
                "repeat_time_seconds": interval,
                "duration": None,
                "task_id": task_id,
                "response_items": {
                    "object_names": [],
                    "location": False,
                    "images": {
                        "visualization": False,
                        "n_z": True,
                        "n_xy": False,
                        "n_xyz": False,
                        "raw": False
                    }
                }
            }
        }

        await self.websocket.send(json.dumps(request))

    async def stop_continuous_data(self, task_id):
        if not self.websocket:
            raise ConnectionError("Not connected to server")

        request = {
            "action": STOP_CONTINUOUS_ACTION,
            "request_data": {
                "task_id": task_id
                }
            }

        await self.websocket.send(
            json.dumps(request)
        )

    async def receive_messages(self):
        if not self.websocket:
            raise ConnectionError("Not connected to server")

        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                if data["response_code"] == DATA_RESPONSE_CODE:
                    self.data_responses_received += 1
                print(f"Received messsage: {data}")
                print('--------------------------------------------------')

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Error in receive_messages: {e}")

    async def close(self):
        if self.websocket:
            print(f'Total data responses received: {self.data_responses_received}')
            await self.websocket.close()
            self.websocket = None


async def main():
    client = DataClient()
    await client.connect()

    try:
        # Start receiving messages in background
        receive_task = asyncio.create_task(client.receive_messages())

        # print("\nStarting continuous data location (0.5-second interval)...")
        # await client.start_continuous_data_location("location_task", interval=0.5)

        await client.get_cars()
        # Let it run for 3 seconds
        await asyncio.sleep(6)

        await client.get_cars()

        await asyncio.sleep(3)

        await client.get_cars()

        # # Stop continuous data
        # print("\nStopping continuous data...")
        # await client.stop_continuous_data("location_task")

        # Wait for stop confirmation and last messages
        await asyncio.sleep(4)

    finally:
        # Clean up
        if 'receive_task' in locals():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
