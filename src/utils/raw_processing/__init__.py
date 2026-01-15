from configs.data_pixel_tensor import DATA_PIXEL_TENSOR_BACKEND

if DATA_PIXEL_TENSOR_BACKEND == "torch":
    try:
        from .utils_raw_processing_torch import *
    except ImportError as err:
        raise ImportError(f"Couldn't use torch as the DATA_PIXEL_TENSOR backend because of: {err}")

elif DATA_PIXEL_TENSOR_BACKEND == "numpy":
    try:
        from .utils_raw_processing_numpy import *
    except ImportError as err:
        raise ImportError(f"Couldn't use numpy as the DATA_PIXEL_TENSOR backend because of: {err}")
