import json
from typing import List, Dict, Any, Union, Literal
from pydantic import BaseModel, ValidationError
from loguru import logger

from deployment.project_la.constants import DATA_RESPONSE_CODE, STATUS_RESPONSE_CODE


class Location(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    X: float
    Y: float
    Z: float


# ImageData now includes all n_ fields again as per the latest screenshot
class ImageData(BaseModel):
    n_z: str
    n_xy: str
    n_xyz: str
    raw: str
    visualized: str


class DetectedObject(BaseModel):
    name: str
    id: int
    location: Location
    timestamp: float
    additional_data: Dict[str, Any]


class DataPayload(BaseModel):
    detected_objects: List[DetectedObject]
    images: ImageData


# Model for "data response" (response_code = 0)
class DetectionDataResponse(BaseModel):
    response_code: Literal[0]  # Discriminator field set to 0
    data: DataPayload          # Contains the detailed detection data


# Model for "status response" (response_code = 1)
class SimpleStatusResponse(BaseModel):
    response_code: Literal[1]  # Discriminator field set to 1
    status: str                # Either "success" or "error"
    data: str                 # Error message or status description


# Create a Union of all possible response types, discriminated by 'response_code'
AnyResponse = Union[DetectionDataResponse, SimpleStatusResponse]

# Optional: If using Pydantic v2, a TypeAdapter can simplify parsing Unions
# ResponseAdapter = TypeAdapter(AnyResponse)

# --- How to Decode ---


def decode_response(raw_json_string: str):
    """
    Decodes a raw JSON string into the appropriate Pydantic model
    based on the 'response_code'. Uses Pydantic's discriminated union.
    """
    try:
        # Pydantic v2 with TypeAdapter (preferred)
        # parsed_response = ResponseAdapter.validate_json(raw_json_string)

        # Manual parsing works for both Pydantic v1 and v2
        data_dict = json.loads(raw_json_string)
        response_code = data_dict.get("response_code")  # Check the code first

        if response_code == DATA_RESPONSE_CODE:
            parsed_response = DetectionDataResponse.model_validate_json(data_dict)
        elif response_code == STATUS_RESPONSE_CODE:
            parsed_response = SimpleStatusResponse.model_validate_json(data_dict)
        else:
            logger.error(f"Error: Unknown or missing response_code '{response_code}'")
            return None

        return parsed_response

    except json.JSONDecodeError:
        logger.error("Error: Invalid JSON string")
        return None
    except ValidationError as e: # Catch Pydantic validation errors
        logger.error(f"Error validating response: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None
