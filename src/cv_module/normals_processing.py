import numpy as np
import cv2


class NormalsProcessor:
    """
    Read a PXI file and calculate the so-called normals nz, nxy, nxyz
    """
    def __init__(self) -> None:
        self.min_scale = 0
        self.max_scale = 1
        self.dolp_factor = 0.05
        self.scale_factor = 1.8

    @staticmethod
    def read_pxi(path):
        with open(path, "rb") as f:
            img = f.read()
        img = np.frombuffer(img, dtype=np.uint8)[64:-68]
        img = img.reshape((2048, 2448))

        return img

    @staticmethod
    def demosaicing_polarization_np(img_bayer):
        # print(img_bayer.dtype)
        img_090 = cv2.resize(img_bayer[::2,::2], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST_EXACT)
        img_090 = np.roll(img_090, (-1,-1), axis=(0,1))

        img_045 = cv2.resize(img_bayer[::2,1::2], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST_EXACT)
        img_045 = np.roll(img_045, (-1,), axis=(0,))

        img_135 = cv2.resize(img_bayer[1::2,::2], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST_EXACT)
        img_135 = np.roll(img_135, (-1,), axis=(1,))

        img_000 = cv2.resize(img_bayer[1::2,1::2], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST_EXACT)
        img_000[:,:] = img_000[:,:]

        i0, i45, i90, i135 = img_000.astype(np.float32), img_045.astype(np.float32), img_090.astype(np.float32), img_135.astype(np.float32)
        return i0, i45, i90, i135

    def get_normals(self, img, ui=False, nz=False, nxy=False, nxyz=False):

        u_i = None
        n_z = None
        n_xy = None
        n_xyz = None

        if ui or nz or nxy or nxyz:
            i0, i45, i90, i135 = self.demosaicing_polarization_np(img)
            s0 = i0 + i90

        if nz or nxy or nxyz:
            s1 = i90 - i0
            s2 = i45 - i135

        if nxy or nxyz:
            theta = 0.5 * np.arctan2(s2, -s1)
        if nz or nxyz:
            dolp = (s1**2 + s2**2)**0.5 / s0

        #######################

        if ui:
            u_i = np.stack([s0, s0, s0], axis=2)
            u_i = np.clip(u_i, 0.0, 255.0).astype(np.uint8)

        #######################

        if nz:
            n_z = dolp * self.scale_factor
            n_z = np.clip(n_z, 0.0, 1.0)
            n_z = (n_z * 255).astype(np.uint8)
            n_z = cv2.cvtColor(n_z, cv2.COLOR_GRAY2BGR)

        ########################

        if nxy:
            theta_deg = ((theta * 180 / np.pi) + 90) / 180
            n_xy = (theta_deg * 255).astype(np.uint8)
            n_xy = cv2.applyColorMap(n_xy, cv2.COLORMAP_JET)

        #########################

        if nxyz:
            sin_d2 = np.sin(dolp)**2
            sin_t2 = np.sin(theta)**2

            x2 = sin_d2 * sin_t2
            y2 = sin_d2 - x2
            z2 = (1-sin_d2) * self.dolp_factor**2

            norm = x2+y2+z2
            x2, y2, z2 = [item / norm for item in [x2, y2, z2]]
            x, y, z = [np.sqrt(item) for item in [x2, y2, z2]]

            n_xyz = np.stack([z, y, x], axis=2)
            n_xyz = (n_xyz * 255).astype(np.uint8)

        return u_i, n_z, n_xy, n_xyz
