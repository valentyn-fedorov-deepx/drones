import numpy as np
from typing import List, Dict
import base64
import cv2

from src.data_pixel_tensor import DataPixelTensor
from src.cv_module.basic_object import BasicObjectWithDistance


def encode_image_to_base64(
    image: np.ndarray,
    format: str = '.png',
    normalize: bool = True
) -> str:
    """
    Encode a numpy array image to base64 string.

    Parameters:
    -----------
    image : np.ndarray
        Input image as a numpy array. Can be either:
        - 2D array (grayscale)
        - 3D array with shape (height, width, channels)
        The array can be either uint8 or float32/float64

    format : str, optional (default='.png')
        Image format to encode to. Common options: '.png', '.jpg', '.jpeg'
        PNG is lossless but larger, JPEG is lossy but smaller

    normalize : bool, optional (default=True)
        If True, automatically normalizes float arrays to [0, 255] range

    Returns:
    --------
    str
        Base64 encoded string of the image

    Raises:
    -------
    ValueError
        If image format is invalid or image processing fails
    TypeError
        If input is not a numpy array
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Make a copy to avoid modifying the original array
    img = image.copy()

    # Handle float arrays
    if img.dtype in [np.float32, np.float64]:
        if normalize:
            # Normalize to [0, 255] range
            img = ((img - img.min()) * (255.0 / (img.max() - img.min()))).astype(np.uint8)
        else:
            # Assume the array is already in [0, 1] range
            img = (img * 255).astype(np.uint8)

    # Ensure format starts with dot
    if not format.startswith('.'):
        format = '.' + format

    # Validate format
    valid_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    if format.lower() not in valid_formats:
        raise ValueError(f"Unsupported format. Must be one of: {valid_formats}")

    try:
        # Encode image to bytes
        success, buffer = cv2.imencode(format, img)
        if not success:
            raise ValueError("Failed to encode image")

        # Convert to base64 string
        base64_string = base64.b64encode(buffer).decode('utf-8')
        return base64_string

    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")


def generate_response_dict(data_tensor: DataPixelTensor,
                           detected_objects: List[BasicObjectWithDistance],
                           response_items: Dict):
    return_names = set(response_items['object_names'])
    send_location = response_items.get("location", False)
    if not return_names:
        filtered_objects = detected_objects
    else:
        filtered_objects = [detected_object for detected_object in detected_objects if detected_object.name in return_names]

    detected_objects_response = list()
    for detected_object in filtered_objects:
        detected_object_response_data = dict(id=detected_object.id,
                                             additional_data=None)
        if send_location:
            detected_object_response_data["longitude"] = detected_object.meas.longitude
            detected_object_response_data["latitude"] = detected_object.meas.latitude
            detected_object_response_data["altitude"] = detected_object.meas.altitude

        detected_objects_response.append(detected_object_response_data)

    images_in_response = dict()

    for data_tensor_element_name, requested in response_items['images'].items():
        if requested:
            tensor = data_tensor[data_tensor_element_name]
            encoded_tensor = encode_image_to_base64(tensor)
            images_in_response[data_tensor_element_name] = encoded_tensor

    full_response = dict(detected_objects=detected_objects_response,
                         images=images_in_response)

    return full_response
