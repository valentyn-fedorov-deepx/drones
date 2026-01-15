from typing import List

from src.cv_module.visualization import plot_object_with_distance
from src.data_pixel_tensor import DataPixelTensor
from src.cv_module.basic_object import BasicObjectWithDistance
from src.utils.image_encode import encode_numpy_array_to_jpg_base64
from deployment.project_la.request_models import ResponseItemsRequest


def prepare_data_response(detected_objects: List[BasicObjectWithDistance],
                          data_pixel_tensor: DataPixelTensor,
                          requested_info: ResponseItemsRequest):
    requested_object_names = requested_info.object_names
    location_requested = requested_info.location
    filtered_detected_objects = []
    filtered_detected_objects_dicts = []
    if requested_object_names:
        for detected_object in detected_objects:
            if detected_object.name in requested_object_names:
                filtered_detected_objects.append(detected_object)
                if location_requested and detected_object.meas is not None:
                    filtered_detected_objects_dicts.append(detected_object.to_project_la_dict())
    else:
        filtered_detected_objects = detected_objects

    images = dict()
    if requested_info.images is not None:
        requested_images = dict(requested_info.images)
        for image_name, image_was_requested in requested_images.items():
            if image_was_requested:
                if image_name == "visualized":
                    visualization = plot_object_with_distance(data_pixel_tensor["view_img"],
                                                              filtered_detected_objects,
                                                              data_pixel_tensor.n_xyz)
                    images[image_name] = encode_numpy_array_to_jpg_base64(visualization, scale=0.5)
                else:
                    # images[image_name] = encode_numpy_array_to_base64(data_pixel_tensor[image_name])
                    images[image_name] = encode_numpy_array_to_jpg_base64(data_pixel_tensor[image_name],
                                                                          scale=0.5)

    response_data = dict(detected_objects=filtered_detected_objects_dicts,
                         images=images)

    return response_data
