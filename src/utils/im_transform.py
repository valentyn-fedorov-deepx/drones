import cv2
from PIL import Image


def resize_with_reflect_padding(image, target_size, border_type=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR):
    """
    Resize a numpy array image to target dimensions while maintaining aspect ratio.
    Uses reflective padding to fill extra space.

    Args:
        image (np.ndarray): Input image array (H, W) or (H, W, C)
        target_size (tuple): Target dimensions as (height, width)
        border_type: Type of reflective border. Options:
                    - cv2.BORDER_REFLECT_101 (default): Reflects without repeating edge pixels
                    - cv2.BORDER_REFLECT: Reflects with repeating edge pixels
                    - cv2.BORDER_WRAP: Wraps around to opposite edge
                    - cv2.BORDER_REPLICATE: Replicates edge pixels
        interpolation: OpenCV interpolation method. Default is cv2.INTER_LINEAR

    Returns:
        np.ndarray: Resized image with reflective padding
    """
    target_height, target_width = target_size
    original_height, original_width = image.shape[:2]

    # Calculate scaling factor to maintain aspect ratio
    scale_factor = min(target_width / original_width, target_height / original_height)

    # Calculate new dimensions after scaling
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    # Calculate padding needed
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left

    # Apply reflective padding using copyMakeBorder
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top, pad_bottom, pad_left, pad_right,
        border_type
    )

    return padded_image

def resize_with_padding(image, target_size, fill_color=0):
    """
    Resize a PIL image to target dimensions while maintaining aspect ratio.
    Uses padding to fill extra space.
    
    Args:
        image (PIL.Image): Input image to resize
        target_size (tuple): Target dimensions as (width, height)
        fill_color (int or tuple): Color for padding. Default is 0 (black).
                                  For RGB images, use tuple like (0, 0, 0)
    
    Returns:
        PIL.Image: Resized image with padding
    """
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # Calculate scaling factor to maintain aspect ratio
    scale_factor = min(target_width / original_width, target_height / original_height)
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and fill color
    # Use same mode as original image
    padded_image = Image.new(image.mode, target_size, fill_color)
    
    # Calculate position to center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Paste the resized image onto the padded image
    padded_image.paste(resized_image, (x_offset, y_offset))
    
    return padded_image
