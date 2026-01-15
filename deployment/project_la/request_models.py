from pydantic import BaseModel
from typing import List, Optional, Literal
from enum import Enum

from deployment.project_la.constants import GET_ACTION, CONTINUOUS_ACTION, STOP_CONTINUOUS_ACTION, SETUP_LOCATION_ACTION, SETUP_CAMERA_ACTION


class ImagesRequest(BaseModel):
    """Schema for images request configuration."""
    visualized: Optional[bool] = False
    n_z: Optional[bool] = False
    n_xy: Optional[bool] = False
    n_xyz: Optional[bool] = False
    raw: Optional[bool] = False


class ResponseItemsRequest(BaseModel):
    """Schema for response items configuration."""
    object_names: List[str]
    location: bool = False
    images: Optional[ImagesRequest] = None


class GetRequest(BaseModel):
    """Schema for a GET request."""
    action: Literal["get"]
    request_data: BaseModel


class GetRequestData(BaseModel):
    """Schema for GET request data."""
    response_items: ResponseItemsRequest


class SetupLocationRequest(BaseModel):
    """Schema for a setup_location request."""
    action: Literal["setup_location"]
    request_data: BaseModel


class SetupLocationRequestData(BaseModel):
    """Schema for setup_location request data."""
    latitude: float
    longitude: float
    altitude: float
    heading: float


class ExposureMode(str, Enum):
    """Valid exposure modes."""
    CONSTANT = "constant"
    OFF = "off"
    CONTINUOS = "continuous"
    ONCE = "once"


class SetupCameraRequest(BaseModel):
    """Schema for a setup_camera request."""
    action: Literal["setup_camera"]
    request_data: BaseModel


class SetupCameraRequestData(BaseModel):
    """Schema for setup_camera request data."""
    exposure_mode: ExposureMode
    exposure_value: Optional[int] = None
    gain_value: Optional[int] = None
    focal_length_mm: Optional[int] = None
    pixel_format: Optional[int] = None


class ContinuousRequest(BaseModel):
    """Schema for a continuous request."""
    action: Literal["continuous"]
    request_data: BaseModel


class ContinuousRequestData(BaseModel):
    """Schema for continuous request data."""
    repeat_time_seconds: Optional[float]
    duration: Optional[float]
    task_id: str
    response_items: ResponseItemsRequest


# Complete Model Definitions with request_data included
class CompleteGetRequest(BaseModel):
    """Complete schema for GET request."""
    action: Literal["get"]
    request_data: GetRequestData


class CompleteSetupLocationRequest(BaseModel):
    """Complete schema for setup_location request."""
    action: Literal["setup_location"]
    request_data: SetupLocationRequestData


class CompleteSetupCameraRequest(BaseModel):
    """Complete schema for setup_camera request."""
    action: Literal["setup_camera"]
    request_data: SetupCameraRequestData


class CompleteContinuousRequest(BaseModel):
    """Complete schema for continuous request."""
    action: Literal["continuous"]
    request_data: ContinuousRequestData


class StopContinuousRequestData(BaseModel):
    """Schema for stop_continuous request data."""
    task_id: str


class StopContinuousRequest(BaseModel):
    """Schema for a stop_continuous request."""
    action: Literal["stop_continuous"]
    request_data: StopContinuousRequestData


# Example usage
def validate_request(json_data: dict):
    """
    Validate a request based on its action type.

    Args:
        json_data: Dictionary containing the request data

    Returns:
        Validated model instance if valid

    Raises:
        ValueError: If validation fails
    """
    action = json_data.get("action")

    if action == GET_ACTION:
        return CompleteGetRequest(**json_data)
    elif action == SETUP_LOCATION_ACTION:
        return CompleteSetupLocationRequest(**json_data)
    elif action == SETUP_CAMERA_ACTION:
        return CompleteSetupCameraRequest(**json_data)
    elif action == CONTINUOUS_ACTION:
        return CompleteContinuousRequest(**json_data)
    elif action == STOP_CONTINUOUS_ACTION:
        return StopContinuousRequest(**json_data)
    else:
        raise ValueError(f"Unknown action type: {action}")


# Example usage with error handling
def validate_request_with_errors(json_data: dict):
    """
    Validate a request and catch any validation errors.

    Args:
        json_data: Dictionary containing the request data

    Returns:
        Tuple of (is_valid: bool, result: object or error message)
    """
    try:
        result = validate_request(json_data)
        return True, result
    except Exception as e:
        return False, str(e)


# Sample code to validate each request type
if __name__ == "__main__":
    import json

    # Example validating get request
    with open("get_request.json", "r") as f:
        get_data = json.load(f)

    is_valid, result = validate_request_with_errors(get_data)
    print(f"Get request valid: {is_valid}")
    if is_valid:
        print(f"Validated data: {result}")
    else:
        print(f"Error: {result}")

    # Example validating stop continuous request
    with open("cancel_continuous_request.json", "r") as f:
        stop_data = json.load(f)

    is_valid, result = validate_request_with_errors(stop_data)
    print(f"Stop continuous request valid: {is_valid}")
    if is_valid:
        print(f"Validated data: {result}")
    else:
        print(f"Error: {result}")
