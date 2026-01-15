from typing import Optional
from datetime import datetime
import time
import struct
from typing import Dict

import numpy as np
import cv2

from src.utils.raw_processing import (demosaicing_polarization, get_stokes,
                                      get_normals, colorize, demosaicing_color,
                                      copy_element, change_dtype, dstack,
                                      custom_jet_colormap,
                                      decode_mono12_contiguous_byteswapped)
from configs.data_pixel_tensor import (PXI_VERSION, ELEMENTS_NAMES,
                                       ELEMENTS_NAMES_WITH_RAW,
                                       NORMALS_NAMES,
                                       POLARIZATION_ELEMENTS_NAME,
                                       STOKES_COMPONENTS,
                                       POLARIZATION_DATA_TYPES,
                                       POLARIZATION_DATA_TYPES_INV,
                                       HEADER_SIZE, DATA_PIXEL_TENSOR_BACKEND)

UNPACKING_METHODS = {
    1: decode_mono12_contiguous_byteswapped
}



class DataPixelTensor:
    """
    A data structure for handling polarization image data with lazy, on-demand
    calculation of derived elements like Stokes parameters and normals.

    Attributes are calculated only when first accessed and the results are cached.
    """
    def __init__(self, raw: np.ndarray, for_display: bool = True,
                 color_data: bool = False, lazy_calculations: bool = True,
                 bit_depth: Optional[int] = None, color_data_main_channel: int = 1,
                 dolp_factor: float = 0.05, scale_factor: float = 1.0,
                 unpack_method: int = 0, height: int = None, width: int = None,
                 created_at: int = None, fake_normals: bool = False):
        """
        Initializes the tensor with raw data and configuration.

        Args:
            raw: The raw image data as a NumPy array.
            for_display: If True, calculated elements are converted to a displayable
                         data type (e.g., uint8) and range.
            color_data: Flag indicating if the raw data is color (Bayer).
            lazy_calculations: If True, all elements of DataPixelTensor will be calculated during the creation
            bit_depth: The bit depth of the raw data. If None, it's inferred.
            color_data_main_channel: The main channel to use for polarization if color_data is True.
            dolp_factor: Factor used in normal calculations.
            scale_factor: Scale factor used in normal calculations.
        """
        if created_at is None:
            self.created_at_datetime = datetime.now()
            self.created_at = self.created_at_datetime.timestamp()
        else:
            self.created_at = created_at
            self.created_at_datetime = datetime.fromtimestamp(self.created_at)

        self.color_data = color_data
        self.color_data_main_channel = color_data_main_channel
        self.for_display = for_display
        self.dolp_factor = dolp_factor
        self.scale_factor = scale_factor
        self.unpack_method = unpack_method

        self._raw = raw

        # Cache for memorizing calculated attributes.
        # It stores the final, public-facing values.
        self._cache: Dict[str, np.ndarray] = {}
        # Cache for intermediate float values needed for calculations.
        self._float_cache: Dict[str, np.ndarray] = {}

        if fake_normals:
            for element_name in ELEMENTS_NAMES_WITH_RAW:
                self._cache[element_name] = raw

        if unpack_method > 0:
            self.packed_data = True
            if bit_depth is None:
                raise ValueError("For packed data you need to specify bit depth")
            if bit_depth == 8:
                self.im_dtype = "uint8"
            elif bit_depth > 8 and bit_depth <= 16:
                self.im_dtype = "uint16"
            else:
                raise ValueError(f"Incorect bit depth: {bit_depth}")

            if height is None or width is None:
                raise ValueError("For packed data you need to specify both height and width")
            self.height, self.width = height, width
            self.bands = 1
            self.data_dtype = "uint8"  # For correctly decoding the image bytes. May require different value for different unpacking schemes
        else:
            self.packed_data = False
            self.im_dtype = self._raw.dtype.name
            self.data_dtype = self.im_dtype
            self.height, self.width = self._raw.shape[:2]
            self.bands = self._raw.shape[2] if self._raw.ndim == 3 else 1

        if bit_depth is None:
            self.bit_depth = np.iinfo(raw.dtype).bits
        else:
            self.bit_depth = bit_depth
        self.max_value = (2 ** self.bit_depth) - 1

        if not lazy_calculations:
            self.calculate_all()

    def calculate_all(self):
        self._process_raw()
        self._calculate_normals()
        self._calculate_view_img()
        self._calculate_visualized_data()

    def convert_to_numpy(self) -> None:
        def _set_numpy(name: str, arr):
            if DATA_PIXEL_TENSOR_BACKEND == "torch":
                if not isinstance(arr, np.ndarray):
                    arr = arr.cpu().numpy()
            self._cache[name] = arr

        for key, val in list(self._cache.items()):
            # Only convert numpy arrays
            _set_numpy(key, val)

    def convert_to_uint8(self) -> None:
        """
        Convert all public-facing elements (including raw) to uint8 in place and
        update the tensor's metadata accordingly.

        Notes:
        - If data was packed, this will decode first (via _process_raw) and then
          store/serve uint8 data going forward (packed flags are cleared).
        - Float intermediates in _float_cache are left as-is (they're internal).
        """
        # Already uint8? Nothing to do.
        if self.im_dtype == "uint8" and self.data_dtype == "uint8" and self.bit_depth == 8:
            return

        # Helper to convert and also reflect to attribute if it exists
        def _set_uint8(name: str, arr: np.ndarray):
            arr = arr / self.max_value
            arr = arr * 255
            arr_u8 = change_dtype(arr, "uint8")
            self._cache[name] = arr_u8

        # 1) Convert cached public elements (whatever has been computed so far)
        for key, val in list(self._cache.items()):
            # Only convert numpy arrays
            _set_uint8(key, val)

        # 2) Ensure 'raw' exists and is converted; also mirror into self._raw
        # if 'raw' not in self._cache:
        #     self._cache['raw'] = self._raw.copy()
        # _set_uint8('raw', self._cache['raw'])

        # 3) Update metadata to reflect uint8 domain (not packed anymore)
        self.im_dtype = "uint8"
        self.data_dtype = "uint8"
        self.bit_depth = 8
        self.max_value = 255
        # self.unpack_method = 0
        # self.packed_data = False

    def _get_raw_for_processing(self) -> np.ndarray:
        """
        Gets the appropriate raw image for polarization calculations,
        performing color demosaicing if necessary.
        """
        self._process_raw()

        if 'processed_raw' not in self._float_cache:
            if self.color_data:
                self._float_cache['processed_raw'] = demosaicing_color(self.raw)[self.color_data_main_channel]
            else:
                self._float_cache['processed_raw'] = self.raw
        return self._float_cache['processed_raw']

    def _calculate_polarization(self):
        """Calculates and caches the four polarization channels."""
        if 'i0' in self._cache:
            return

        self._process_raw()
        raw_image = self._get_raw_for_processing()
        i0, i45, i90, i135 = demosaicing_polarization(raw_image)

        self._float_cache.update({'i0': i0, 'i45': i45, 'i90': i90, 'i135': i135})

        if self.for_display:
            self._cache['i0'] = change_dtype(i0, self.im_dtype)
            self._cache['i45'] = change_dtype(i45, self.im_dtype)
            self._cache['i90'] = change_dtype(i90, self.im_dtype)
            self._cache['i135'] = change_dtype(i135, self.im_dtype)
        else:
            self._cache.update(self._float_cache)

    def _calculate_stokes(self):
        """Calculates and caches the three Stokes parameters."""
        if 's0' in self._cache:
            return

        self._calculate_polarization()

        s0, s1, s2 = get_stokes(self._float_cache['i0'], self._float_cache['i45'],
                                self._float_cache['i90'], self._float_cache['i135'])

        self._float_cache.update({'s0': s0, 's1': s1, 's2': s2})

        if self.for_display:
            self._cache['s0'] = change_dtype(s0 / 2, self.im_dtype)
            self._cache['s1'] = change_dtype((s1 + self.max_value / 2) / 2, self.im_dtype)
            self._cache['s2'] = change_dtype((s2 + self.max_value / 2) / 2, self.im_dtype)
        else:
            self._cache['s0'] = s0
            self._cache['s1'] = s1
            self._cache['s2'] = s2

    def _process_raw(self):
        if 'raw' in self._cache:
            return

        if self.packed_data:
            method = UNPACKING_METHODS[self.unpack_method]
            raw = method(self._raw, self.width, self.height)
        else:
            raw = self._raw.copy()
        self._cache['raw'] = raw

    def _calculate_normals(self):
        """Calculates and caches the surface normals."""
        if 'n_z' in self._cache:
            return

        self._calculate_stokes()

        n_z, n_xy, n_xyz = get_normals(self._float_cache['s1'], self._float_cache['s2'], self._float_cache['s0'],
                                       self.scale_factor, self.dolp_factor)

        self._float_cache.update({'n_z': n_z, 'n_xy': n_xy, 'n_xyz': n_xyz})

        if self.for_display:
            self._cache['n_z'] = change_dtype((n_z * self.max_value).round(), self.im_dtype)
            n_xy = custom_jet_colormap(n_xy)
            n_xy = change_dtype((n_xy * self.max_value).round(), self.im_dtype)

            self._cache['n_xy'] = n_xy
            self._cache['n_xyz'] = change_dtype((n_xyz * self.max_value).round(), self.im_dtype)
        else:
            self._cache['n_z'] = n_z
            self._cache['n_xy'] = n_xy
            self._cache['n_xyz'] = n_xyz

    def _calculate_view_img(self):
        """Calculates and caches the main color view image."""
        if 'view_img' in self._cache:
            return

        self._process_raw()

        if self.color_data:
            view_image = colorize(self.raw, self.max_value, self.for_display)
        else:
            self._calculate_stokes()

            s0_float = self._float_cache['s0']
            view_image = dstack([s0_float, s0_float, s0_float]) / 2
            view_image = change_dtype(view_image, self.im_dtype)

        self._cache['view_img'] = view_image

    def _calculate_visualized_data(self):
        """Calculates and caches the data for visualization."""
        if 'visualized_data' in self._cache:
            return

        self._calculate_view_img()
        self._cache['visualized_data'] = copy_element(self._cache['view_img'])

    def __getattr__(self, name: str) -> np.ndarray:
        """
        Dynamically calculates and returns an attribute on first access.
        This method is the core of the lazy calculation logic.
        """
        if name not in ELEMENTS_NAMES + ['raw']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check cache first. If it's there, it means it was calculated by a
        # dependency call, but not yet set on the instance.
        if name in self._cache:
            value = self._cache[name]
            setattr(self, name, value)
            return value

        # --- Dispatch to the correct calculation group ---
        if name in POLARIZATION_ELEMENTS_NAME:
            self._calculate_polarization()
        elif name in STOKES_COMPONENTS:
            self._calculate_stokes()
        elif name in NORMALS_NAMES:
            self._calculate_normals()
        elif name == 'view_img':
            self._calculate_view_img()
        elif name == 'visualized_data':
            self._calculate_visualized_data()
        elif name == 'raw':
            self._process_raw()

        # The attribute should now be in the cache. Retrieve it,
        # set it on the instance to prevent future __getattr__ calls for it,
        # and return it.
        try:
            value = self._cache[name]
            setattr(self, name, value)
            return value
        except KeyError:
            raise AttributeError(f"Logic error: Could not calculate attribute '{name}'")

    def __getitem__(self, item_name: str) -> np.ndarray:
        """Provides dictionary-style access to elements."""
        if item_name not in ELEMENTS_NAMES_WITH_RAW:
            raise ValueError(f"Data pixel tensor does not contain {item_name}")
        return getattr(self, item_name)

    def copy(self):
        """Creates a fresh copy of the tensor without cached data."""
        return DataPixelTensor(self._raw, self.for_display,
                               self.color_data,
                               bit_depth=self.bit_depth,
                               color_data_main_channel=self.color_data_main_channel,
                               dolp_factor=self.dolp_factor,
                               scale_factor=self.scale_factor,
                               unpack_method=self.unpack_method,
                               height=self.height,
                               width=self.width
                               )

    @staticmethod
    def parse_header_from_buffer(buffer_data, header_size=None):
        """
        Parse header from numpy buffer data.

        Args:
            buffer_data (numpy.ndarray): Buffer data as numpy array of uint8
            header_size (int, optional): Size of header in bytes. If None, uses HEADER_SIZE constant.

        Returns:
            dict: Parsed header containing all metadata fields
        """
        # Use the constant or provided header size
        if header_size is None:
            header_size = HEADER_SIZE

        # Extract header bytes
        header_bytes = buffer_data[:header_size]

        # Define format string for unpacking
        format_string = '<' + 'i' * 9 + 'q' + 'i' * 5

        # Unpack the binary data
        unpacked_data = struct.unpack(format_string, header_bytes)

        # Create header dictionary
        header = {
            'pxi_version': unpacked_data[0],
            'polarization_data_type': unpacked_data[1],
            'width': unpacked_data[2],
            'height': unpacked_data[3],
            'bit_depth': unpacked_data[5],
            'unpacking_method': unpacked_data[6],
            'is_scaled': unpacked_data[7],
            'color_data': unpacked_data[8],
            'created_at': unpacked_data[9],
            'number_of_metadata': unpacked_data[10],
            'is_non_bayer': unpacked_data[11],
        }

        return header

    @classmethod
    def from_pxi_file(cls, file_path, **kwargs):
        """
        Create instance from PXI file using the static header parser.
        """
        with open(file_path, "rb") as f:
            img = f.read()

        data = np.frombuffer(img, dtype=np.uint8)

        # Use the static method to parse header
        header = cls.parse_header_from_buffer(data)

        dtype = np.dtype(POLARIZATION_DATA_TYPES_INV[header['polarization_data_type']])
        img_is_packed = header['unpacking_method'] > 0
        im_size = header['height'] * header['width']

        if img_is_packed:
            im_size = int(im_size * (header['bit_depth'] / np.iinfo(dtype).bits))

        header_end = HEADER_SIZE // dtype.itemsize
        image_end = im_size + header_end

        # Set kwargs from header
        if "color_data" not in kwargs:
            kwargs["color_data"] = header["color_data"]
        if header['bit_depth']:
            kwargs['bit_depth'] = header['bit_depth']
        kwargs['created_at'] = header['created_at']

        if img_is_packed:
            img_data = img[header_end:image_end]
            kwargs['unpack_method'] = header['unpacking_method']
            kwargs['height'] = header['height']
            kwargs['width'] = header['width']
        else:
            img_data = np.frombuffer(img, dtype=dtype)[header_end:image_end]
            img_data = img_data.reshape((header['height'], header["width"]))

        return cls(img_data, **kwargs)

    @classmethod
    def from_file(cls, file_path: str, **kwargs):
        if file_path.endswith(".pxi"):
            return cls.from_pxi_file(file_path, **kwargs)
        elif file_path.endswith(".npy"):
            return cls.from_npy_file(file_path, **kwargs)
        elif file_path.endswith(".png") or file_path.endswith(".tif"):
            return cls.from_img_file(file_path, **kwargs)
        else:
            raise ValueError(f"Received file with unsupported extension: {file_path}")

    @classmethod
    def from_img_file(cls, file_path: str, **kwargs):
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        return cls(image, **kwargs)

    @classmethod
    def from_npy_file(cls, file_path: str, **kwargs):
        image = np.load(str(file_path))
        return cls(image, **kwargs)

    def save_pxi(self, path: str):
        # This method remains the same as in the original code
        is_scaled = False
        number_of_metadata = 0
        isNonBayer = not self.color_data
        binary_header = struct.pack('<' + 'i' * 9 + 'q' + 'i' * 5,
                                    PXI_VERSION,
                                    POLARIZATION_DATA_TYPES[self.data_dtype],
                                    self.width,
                                    self.height,
                                    self.bands,
                                    self.bit_depth, self.unpack_method,
                                    is_scaled,
                                    self.color_data,
                                    int(self.created_at),
                                    number_of_metadata,
                                    isNonBayer,
                                    0, 0, 0)
        image_encoded = self._raw.tobytes()
        binary_data = binary_header + image_encoded
        with open(path, 'wb') as f:
            f.write(binary_data)


if __name__ == "__main__":
    im_path = "/home/jetson/dev/dx_vyzai_python/frame_2.png"

    raw_img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    data_tensor = DataPixelTensor(raw_img, True, lazy_calculations=False)
