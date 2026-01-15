import socket

def get_local_ip():
    """
    Get the local IP address of the device.
    
    Returns:
        str: The local IP address.
    """
    try:
        # Create a socket that connects to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # It doesn't actually connect but it helps determine which interface to use
        s.connect(("8.8.8.8", 80))
        # Get the local IP address used for this connection
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return "127.0.0.1"  # Return localhost if there's an error