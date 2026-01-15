import struct
import numpy as np
from PIL import Image
import io
from dataclasses import dataclass


def numpy_to_jpg_bytes(image_array):
    """
    Convert a NumPy array to JPEG bytes.

    Parameters:
    -----------
    image_array : numpy.ndarray
        The input image array. Should be in HWC (Height, Width, Channels) format.
        Typically, this is a uint8 array with values between 0-255.

    Returns:
    --------
    bytes
        JPEG-encoded image bytes

    Raises:
    -------
    ValueError
        If the input array is not a NumPy array or has incorrect dimensions
    """
    # Validate input
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Ensure the array is uint8
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    # Create PIL Image from NumPy array
    pil_image = Image.fromarray(image_array)

    # Create a bytes buffer
    byte_arr = io.BytesIO()

    # Save image to the buffer in JPEG format
    pil_image.save(byte_arr, format='JPEG')

    # Get the byte array
    return byte_arr.getvalue()


def jpg_bytes_to_pil_image(jpg_bytes):
    """
    Convert JPEG bytes to a PIL Image object.

    Parameters:
    -----------
    jpg_bytes : bytes
        JPEG-encoded image bytes

    Returns:
    --------
    PIL.Image.Image
        Restored PIL Image object

    Raises:
    -------
    ValueError
        If input is not bytes or cannot be decoded
    """
    # Validate input
    if not isinstance(jpg_bytes, bytes):
        raise ValueError("Input must be bytes")

    try:
        # Create a bytes buffer from the input
        byte_stream = io.BytesIO(jpg_bytes)

        # Open the image from the bytes stream
        pil_image = Image.open(byte_stream)

        # Ensure the image is loaded
        pil_image.load()

        return pil_image

    except Exception as e:
        raise ValueError(f"Failed to decode JPEG bytes: {str(e)}")


@dataclass
class ShotEvent:
    packet_id: int  # uint16_t
    packet_size: int  # uint32_t
    timestamp: int  # uint32_t
    pitch: float  # float32, deegrees
    roll: float  # float32, deegrees
    yaw: float  # float32, deegrees
    accel_x: float  # float32, meters per second per second
    accel_y: float  # float32, meters per second per second
    accel_z: float  # float32, meters per second per second
    shooter_lat: float  # double, deegrees
    shooter_lon: float  # double, deegrees
    shooter_alt: float  # double, meters
    human_detection: int  # uint16_t
    # 0 = not detected (bit mask for body parts)
    # 0x00000000000000001 = HEAD
    # 0x00000000000000010 = NECK
    # 0x00000000000000100 = CHEST
    # 0x00000000000001000 = BACK
    # 0x00000000000010000 = TORSO
    # 0x00000000000100000 = PELVIS
    # 0x00000000001000000 = LEG_LEFT
    # 0x00000000010000000 = LEG_RIGHT
    # 0x00000000100000000 = ARM_LEFT
    # 0x00000001000000000 = ARM_RIGHT
    aim_point_body_part: int  # uint8_t
    # 1 = BODY_PART_HEAD
    # 2 = BODY_PART_NECK
    # 3 = BODY_PART_CHEST
    # 4 = BODY_PART_BACK
    # 5 = BODY_PART_TORSO
    # 6 = BODY_PART_PELVIS
    # 7 = BODY_PART_LEFT_LEG
    # 8 = BODY_PART_RIGHT_LEG
    # 9 = BODY_PART_ARM_LEFT
    # 10 = BODY_PART_ARM_RIGHT
    hit_point_body_part: int  # uint8_t
    # 1 = BODY_PART_HEAD
    # 2 = BODY_PART_NECK
    # 3 = BODY_PART_CHEST
    # 4 = BODY_PART_BACK
    # 5 = BODY_PART_TORSO
    # 6 = BODY_PART_PELVIS
    # 7 = BODY_PART_LEFT_LEG
    # 8 = BODY_PART_RIGHT_LEG
    # 9 = BODY_PART_ARM_LEFT
    # 10 = BODY_PART_ARM_RIGHT
    body_pose: int  # uint8_t
    # 0 = Stand
    # 1 = Kneel
    # 2 = Prone
    # 3 = Profile_Left
    # 4 = Profile_Right
    # Always center of the image
    aim_point_x: int  # uint16_t
    aim_point_y: int  # uint16_t
    aim_point_z: int  # uint16_t
    # Aim point modified by ballistics
    hit_point_x: int  # uint16_t
    hit_point_y: int  # uint16_t
    hit_point_z: int  # uint16_t
    residual_energy: float  # float32
    residual_velocity: float  # float32
    target_range: int  # uint16_t, meter
    target_velocity_x: float  # float32, m/sec
    target_velocity_y: float  # float32, m/sec
    target_velocity_z: float  # float32, m/sec
    occlusion: int  # uint8_t, boolean
    occluding_object: int  # uint8_t,
    # Occluding object enumeration
    # Tree, bush, vehicle, wall, etc.
    object_x: int  # uint16_t
    object_y: int  # uint16_t
    object_z: int  # uint16_t
    object_range: int  # uint16_t, meter
    impact_polygon_shift_x: int  # uint16_t, pixels
    impact_polygon_shift_y: int  # uint16_t, pixels
    impact_polygon_size: int  # uint16_t, bytes
    image_size: int  # uint16_t, bytes
    impact_polygon_string: str  # string
    jpeg_bit_stream: np.ndarray  # image, array of bytes

    def encode_to_bytes(self):
        """
        Encode all attributes of the ShotEvent instance into a byte message.

        The encoding follows the order of attributes as declared in the class definition
        and uses the specified data types:
        - uint16_t for packet_id
        - uint32_t for packet_size and timestamp 
        - float32 for angular and acceleration measurements
        - double for lat, lon, alt
        - Various uint8_t and uint16_t for other attributes
        - Special handling for string and numpy array

        Returns:
            bytes: Encoded message containing all ShotEvent attributes
        """
        # Encode impact_polygon_string as UTF-8 bytes with length prefix
        script_bytes = self.impact_polygon_string.encode('utf-8')
        script_length = len(script_bytes)
        script_length_bytes = struct.pack('<I', script_length)

        # Encode image bit stream (numpy array)
        # First, pack the length of the array
        im_bytes = numpy_to_jpg_bytes(self.jpeg_bit_stream)
        jpeg_length = len(im_bytes)
        jpeg_length_bytes = struct.pack('<I', jpeg_length)
        packetSize = 123 + script_length + jpeg_length

        # import ipdb; ipdb.set_trace()

        # Prepare primitive type attributes for packing
        primitive_data = struct.pack(
            '<' +  # little-endian
            'H' +   # packet_id (uint16_t)
            'I' +   # packet_size (uint32_t)
            'I' +   # timestamp (uint32_t)
            'f' +   # pitch (float32)
            'f' +   # roll (float32)
            'f' +   # yaw (float32)
            'f' +   # accel_x (float32)
            'f' +   # accel_y (float32)
            'f' +   # accel_z (float32)
            'd' +   # shooter_lat (double)
            'd' +   # shooter_lon (double)
            'd' +   # shooter_alt (double)
            'H' +   # human_detection (uint16_t)
            'B' +   # aim_point_body_part (uint8_t)
            'B' +   # hit_point_body_part (uint8_t)
            'B' +   # body_pose (uint8_t)
            'H' +   # aim_point_x (uint16_t)
            'H' +   # aim_point_y (uint16_t)
            'H' +   # aim_point_z (uint16_t)
            'H' +   # hit_point_x (uint16_t)
            'H' +   # hit_point_y (uint16_t)
            'H' +   # hit_point_z (uint16_t)
            'f' +   # residual_energy (float32)
            'f' +   # residual_velocity (float32)
            'H' +   # target_range (uint16_t)
            'f' +   # target_velocity_x (float32)
            'f' +   # target_velocity_y (float32)
            'f' +   # target_velocity_z (float32)
            'B' +   # occlusion (uint8_t)
            'B' +   # occluding_object (uint8_t)
            'H' +   # object_x (uint16_t)
            'H' +   # object_y (uint16_t)
            'H' +   # object_z (uint16_t)
            'H' +   # object_range (uint16_t)
            'H' +   # impact_polygon_shift_x (uint16_t)
            'H' +   # impact_polygon_shift_y (uint16_t)
            'H' +   # impact_polygon_size (uint16_t)
            'H',    # image_size (uint16_t)
            int(self.packet_id),
            packetSize,
            int(self.timestamp),
            self.pitch,
            self.roll,
            self.yaw,
            self.accel_x,
            self.accel_y,
            self.accel_z,
            self.shooter_lat,
            self.shooter_lon,
            self.shooter_alt,
            int(self.human_detection),
            int(self.aim_point_body_part),
            int(self.hit_point_body_part),
            int(self.body_pose),
            int(self.aim_point_x),
            int(self.aim_point_y),
            int(self.aim_point_z),
            int(self.hit_point_x),
            int(self.hit_point_y),
            int(self.hit_point_z),
            self.residual_energy,
            self.residual_velocity,
            int(self.target_range),
            self.target_velocity_x,
            self.target_velocity_y,
            self.target_velocity_z,
            int(self.occlusion),
            int(self.occluding_object),
            int(self.object_x),
            int(self.object_y),
            int(self.object_z),
            int(self.object_range),
            int(self.impact_polygon_shift_x),
            int(self.impact_polygon_shift_y),
            int(self.impact_polygon_size),
            int(self.image_size)
        )

        # import ipdb; ipdb.set_trace
        # Combine all parts of the message
        return (
            primitive_data +          # Primitive type attributes
            script_length_bytes +     # Length of script string
            script_bytes +            # Script string itself
            jpeg_length_bytes +       # Length of JPEG bit stream
            im_bytes  # JPEG bit stream bytes
        )

    @staticmethod
    def decode_from_bytes(message_bytes):
        """
        Decode bytes back into a ShotEvent instance.

        This is a complementary method to encode_to_bytes() for decoding the message.

        Args:
            message_bytes (bytes): Byte message to decode

        Returns:
            ShotEvent: Decoded instance with attributes restored
        """
        # Unpack primitive type attributes
        primitive_format = '<' + 'H' + 'I' + 'I' + 'f' + 'f' + 'f' + 'f' + 'f' + 'f' + 'd' + 'd' + 'd' + 'H' + 'B' + 'B' + 'B' + 'H' + 'H' + 'H' + 'H' + 'H' + 'H' + 'f' + 'f' + 'H' + 'f' + 'f' + 'f' + 'B' + 'B' + 'H' + 'H' + 'H' + 'H' + 'H' + 'H' + 'H' + 'H'

        primitive_size = struct.calcsize(primitive_format)

        primitive_data = struct.unpack(
            primitive_format,
            message_bytes[:primitive_size]
        )

        # Unpack string and image data
        cursor = primitive_size

        # Decode script string
        script_length = struct.unpack('<I', message_bytes[cursor:cursor+4])[0]
        cursor += 4
        impact_polygon_string = message_bytes[cursor:cursor+script_length].decode('utf-8')
        cursor += script_length

        # Decode image bit stream
        jpeg_length = struct.unpack('<I', message_bytes[cursor:cursor+4])[0]
        cursor += 4
        pil_image = jpg_bytes_to_pil_image(message_bytes[cursor:cursor+jpeg_length])
        jpeg_bit_stream = np.asarray(pil_image)
        # Image.frombuffer(message_bytes[cursor:cursor+jpeg_length])
        # jpeg_bit_stream = np.frombuffer(message_bytes[cursor:cursor+jpeg_length], dtype=np.uint8)

        # Create a new ShotEvent and populate its attributes

        # Assign primitive attributes
        attrs = [
            'packet_id', 'packet_size', 'timestamp',
            'pitch', 'roll', 'yaw',
            'accel_x', 'accel_y', 'accel_z',
            'shooter_lat', 'shooter_lon', 'shooter_alt',
            'human_detection',
            'aim_point_body_part', 'hit_point_body_part', 'body_pose',
            'aim_point_x', 'aim_point_y', 'aim_point_z',
            'hit_point_x', 'hit_point_y', 'hit_point_z',
            'residual_energy', 'residual_velocity',
            'target_range',
            'target_velocity_x', 'target_velocity_y', 'target_velocity_z',
            'occlusion', 'occluding_object',
            'object_x', 'object_y', 'object_z', 'object_range',
            'impact_polygon_shift_x', 'impact_polygon_shift_y',
            'impact_polygon_size', 'image_size'
        ]

        shot_event_data = dict()

        for attr, value in zip(attrs, primitive_data):
            shot_event_data[attr] = value

        shot_event_data["impact_polygon_string"] = impact_polygon_string
        shot_event_data["jpeg_bit_stream"] = jpeg_bit_stream

        decoded_event = ShotEvent(**shot_event_data)

        return decoded_event

    def to_dict(self):
        shot_event_dict = self.__dict__.copy()
        del shot_event_dict['jpeg_bit_stream']
        return shot_event_dict