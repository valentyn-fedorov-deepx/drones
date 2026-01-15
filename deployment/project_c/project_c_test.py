import asyncio
import websockets

from src.project_managers.outputs.project_c_outputs import ShotEvent


SERVER_PORT = 8765
SERVER_HOST = "192.168.4.186"


async def client():
    """
    WebSocket client to connect to the server and send/receive messages.
    """
    try:
        # Establish connection to the WebSocket server
        async with websockets.connect(f'ws://{SERVER_HOST}:{SERVER_PORT}') as websocket:
            # Send a message to the server
            await websocket.send("shot")
            print("Message sent to server")

            # Receive response from the server
            response = await websocket.recv()
            shot_event = ShotEvent.decode_from_bytes(response)
            import cv2
            cv2.imwrite("server_response.png", shot_event.jpeg_bit_stream)
            print(f"Response")

    except Exception as e:
        print(f"An error occurred: {e}")


async def multiple_messages_client():
    """
    Client that sends multiple messages and demonstrates continuous communication.
    """
    try:
        async with websockets.connect(f'ws://localhost:{SERVER_PORT}') as websocket:
            messages = [
                "shot"
            ]

            for msg in messages:
                print(f"Sent")
                await websocket.send(msg)
                response = await websocket.recv()
                print(f"Received")
                shot_event = ShotEvent.decode_from_bytes(response)

                await asyncio.sleep(1)  # Small delay between messages

    except Exception as e:
        print(f"Connection error: {e}")

# Run the client
if __name__ == "__main__":
    # Choose which client function to run
    asyncio.run(client())
    # Uncomment the line below to run multiple messages client instead
    # asyncio.run(multiple_messages_client())
