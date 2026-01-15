# By Oleksiy Grechnyev, 5/6/24

import sys
import pathlib

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
    def demosaicing_polarization(img_bayer):
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
            i0, i45, i90, i135 = self.demosaicing_polarization(img)
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




########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class FrameSource:
    """
    Read a video source (video file or directory with PXI files) and provide frames with
    so-called normals (real or fake)
    """

    def __init__(self, video_path, pos, max_frames):
        self.video_path = video_path
        p_video_path = pathlib.Path(video_path)
        self.pxi_mode = p_video_path.is_dir()
        self.pos = 0 if pos is None else pos
        self.max_frames = max_frames
        
        self.nproc = None
        self.video_in = None
        self.file_list_pxi = None

        self.i_frame = 0

        if not self.pxi_mode:
            self.video_in = cv2.VideoCapture(video_path)
            assert self.video_in.isOpened()
            if self.pos > 0:
                self.video_in.set(cv2.CAP_PROP_POS_FRAMES, self.pos)
            self.fps = self.video_in.get(cv2.CAP_PROP_FPS)
        else:
            # Get PXI file list
            self.file_list_pxi = sorted([p for p in p_video_path.iterdir() if p.suffix.lower() == '.pxi'])
            self.fps = 25
            # Create the normals processor
            self.nproc = NormalsProcessor()

    def __iter__(self):
        return self

    @staticmethod
    def add_tint(img, tint):
        img = img.astype('int16')
        img = (img + tint) // 2
        return img.astype('uint8')

    @staticmethod
    def create_fake_normals(frame0):
        u_i = frame0
        n_z = frame0
        n_xy = FrameSource.add_tint(frame0, [0, 0xff, 0])
        n_xyz = FrameSource.add_tint(frame0, [0, 0, 0xff])
        return u_i, n_z, n_xy, n_xyz

    def __next__(self):
        if self.max_frames is not None and self.i_frame >= self.max_frames:
            raise StopIteration

        if not self.pxi_mode:
            # Read frame, Create fake normals
            ret, frame = self.video_in.read()
            if not ret or frame is None:
                raise StopIteration
            img_i, img_nz, img_nxy, img_nxyz = self.create_fake_normals(frame)
        else:
            # Read PXI, create real normals
            if self.pos >= len(self.file_list_pxi):
                raise StopIteration

            frame_pxi = self.nproc.read_pxi(str(self.file_list_pxi[self.pos]))
            img_i, img_nz, img_nxy, img_nxyz = self.nproc.get_normals(frame_pxi, ui=True, nz=True, nxy=True,
                                                                      nxyz=True)  # ALl  (2048, 2448, 3)

        self.pos += 1
        self.i_frame += 1

        return img_i, img_nz, img_nxy, img_nxyz

    @property
    def pos_sec(self):
        """Position in seconds """
        if not self.pxi_mode:
            return self.video_in.get(cv2.CAP_PROP_POS_MSEC) / 1000
        else:
            return self.pos / self.fps


########################################################################################################################

if __name__ == "__main__":
    import cv2
    from tqdm import tqdm

    pxi_path = pathlib.Path("D:\Projects\deepx\people-track\code_gui\src\shapeos\pxi_sources")
    source = FrameSource(pxi_path, None, None)
    output_path = pathlib.Path("normals")
    output_path.mkdir()

    for i, item in tqdm(enumerate(source)):
        img_i, img_nz, img_nxy, img_nxyz = item

        cv2.imwrite(str(output_path / f"{i}_img_i.png"), img_i)
        cv2.imwrite(str(output_path / f"{i}_img_nz.png"), img_nz)
        cv2.imwrite(str(output_path / f"{i}_img_nxy.png"), img_nxy)
        cv2.imwrite(str(output_path / f"{i}_img_nxyz.png"), img_nxyz)