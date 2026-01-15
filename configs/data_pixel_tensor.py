PXI_VERSION = 1
ELEMENTS_NAMES = ["i0", "i45", "i90", "i135",
                  "s0", "s1", "s2", "n_z",
                  "n_xy", "n_xyz", "view_img",
                  "visualized_data"]

ELEMENTS_NAMES_WITH_RAW = ELEMENTS_NAMES + ['raw']

NORMALS_NAMES = ["n_z", "n_xy", "n_xyz"]
POLARIZATION_ELEMENTS_NAME = ["i0", "i45", "i90", "i135"]
STOKES_COMPONENTS = ["s0", "s1", "s2"]

DISPLAY_NAMES = NORMALS_NAMES + ["view_img", "raw"]

POLARIZATION_DATA_TYPES = {
    "uint8": 1,
    "uint16": 4  # skiped 2 and 3 for compatability with the shapeos
}

POLARIZATION_DATA_TYPES_INV = {idx: data_type for data_type, idx in POLARIZATION_DATA_TYPES.items()}
HEADER_SIZE = 64


DATA_PIXEL_TENSOR_BACKEND = 'torch'
TORCH_DEVICE = "cuda"
# TORCH_DEVICE = "cpu"

DATA_PIXEL_TENSOR_BACKEND = 'numpy'

ENABLE_COLOR_CORRECTION = False
COLOR_CORRECTION_MODEL_PATH = "models/color_model.pkl"
