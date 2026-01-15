import torch
import numpy as np
import cv2


class NormalsProcessorCUDA:
    """The same using pytorch on cuda"""

    def __init__(self):
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
    def demosaicing_polarization(img_bayer):

        img_090 = torch.nn.functional.interpolate(img_bayer[...,::2,::2], scale_factor=2, mode="nearest-exact")
        img_090 = torch.roll(img_090, (-1, -1), dims=(2, 3))

        img_045 = torch.nn.functional.interpolate(img_bayer[...,::2,1::2], scale_factor=2, mode="nearest-exact")
        img_045 = torch.roll(img_045, (-1,), dims=(2,))

        img_135 = torch.nn.functional.interpolate(img_bayer[...,1::2,::2], scale_factor=2, mode="nearest-exact")
        img_135 = torch.roll(img_135, (-1,), dims=(3,))

        img_000 = torch.nn.functional.interpolate(img_bayer[...,1::2,1::2], scale_factor=2, mode="nearest-exact")

        i0, i45, i90, i135 = img_000[0].permute(1, 2, 0), img_045[0].permute(1, 2, 0), img_090[0].permute(1, 2, 0), img_135[0].permute(1, 2, 0)
        return i0, i45, i90, i135

    def get_normals(self, img, ui=False, nz=False, nxy=False, nxyz=False):

        u_i = None
        n_z = None
        n_xy = None
        n_xyz = None

        if ui or nz or nxy or nxyz:
            img = torch.Tensor(img).cuda().float()[None]
            i0, i45, i90, i135 = self.demosaicing_polarization(img.unsqueeze(0))
            s0 = i0 + i90

        if nz or nxy or nxyz:
            s1 = i90 - i0
            s2 = i45 - i135

        if nxy or nxyz:
            theta = 0.5 * torch.arctan2(s2, -s1)
        if nz or nxyz:
            dolp = (s1**2 + s2**2)**0.5 / s0

        #######################

        if ui:
            u_i = torch.cat([s0, s0, s0], dim=2)
            u_i = torch.clip(u_i, 0.0, 255.0)
            u_i = u_i.cpu().numpy().astype(np.uint8)

        #######################

        if nz:
            n_z = dolp * self.scale_factor
            n_z = torch.clip(n_z, 0.0, 1.0)
            n_z = (n_z * 255).cpu().numpy().astype(np.uint8)
            n_z = cv2.cvtColor(n_z, cv2.COLOR_GRAY2BGR)

        ########################

        if nxy:
            theta_deg = ((theta * 180 / torch.pi) + 90) / 180
            n_xy = (theta_deg * 255).cpu().numpy().astype(np.uint8)
            n_xy = cv2.applyColorMap(n_xy, cv2.COLORMAP_JET)

        #########################

        if nxyz:
            sin_d2 = torch.sin(dolp)**2
            sin_t2 = torch.sin(theta)**2

            x2 = sin_d2 * sin_t2
            y2 = sin_d2 - x2
            z2 = (1-sin_d2) * self.dolp_factor**2

            norm = x2+y2+z2
            x2, y2, z2 = [item / norm for item in [x2, y2, z2]]
            x, y, z = [torch.sqrt(item) for item in [x2, y2, z2]]

            n_xyz = torch.cat([z, y, x], axis=2)
            n_xyz = (n_xyz * 255).cpu().numpy().astype(np.uint8)

        return u_i, n_z, n_xy, n_xyz
