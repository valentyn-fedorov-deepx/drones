import base64
from PIL import Image
from io import BytesIO
import cv2


def encode_numpy_array_to_jpg_base64(numpy_image, jpeg_quality: int = 80, scale: float = None):
    if scale is not None and scale != 1.0:
        new_width = int(numpy_image.shape[1] * scale)
        new_height = int(numpy_image.shape[0] * scale)

        # Resize the image
        numpy_image = cv2.resize(numpy_image, (new_width, new_height),
                            interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', numpy_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    img_str = base64.b64encode(buffer).decode('utf-8')

    return img_str


def encode_numpy_array_to_base64(numpy_array, image_format='PNG'):
    """
    Encodes a NumPy array (representing an image) to a Base64 string.

    Args:
        numpy_array (numpy.ndarray): The NumPy array representing the image.
                                     It should be in a format that PIL can understand
                                     (e.g., uint8 arrays with shape (height, width, 3) for RGB,
                                     (height, width) for grayscale).
        image_format (str, optional): The image format to use for encoding.
                                       Common formats are 'PNG' (lossless), 'JPEG' (lossy), 'GIF', etc.
                                       Defaults to 'PNG'.

    Returns:
        str: A Base64 encoded string representing the image.
    """
    try:
        # 1. Convert NumPy array to PIL Image
        image = Image.fromarray(numpy_array).convert("RGB")

        # 2. Save the PIL Image to a BytesIO buffer (in memory)
        image_buffer = BytesIO()
        image.save(image_buffer, format=image_format)
        image_binary = image_buffer.getvalue()

        # 3. Encode the binary image data to Base64
        base64_encoded_data = base64.b64encode(image_binary)

        # 4. Decode the Base64 bytes to a string (UTF-8)
        base64_string = base64_encoded_data.decode('utf-8')

        return base64_string

    except Exception as e:
        print(f"Error encoding NumPy array to Base64: {e}")
        return None
