try:
    import vyz
except ImportError as err:
    print("Couldn't import vyzai module.", err)
    exit(0)

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

from frame_source import FrameSource

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--save-csv", action="store_true", default=False)
    parser.add_argument("--live-feed", action="store_true", default=False)

    return parser.parse_args()


def get_normals(s0, s1, s2):
    theta = 0.5 * np.arctan2(s2, -s1)
    dolp = (s1**2 + s2**2)**0.5 / s0

    u_i = np.stack([s0, s0, s0], axis=2)
    u_i = np.clip(u_i, 0.0, 255.0).astype(np.uint8)

    n_z = dolp * scale_factor
    n_z = np.clip(n_z, 0.0, 1.0)
    n_z = (n_z * 255).astype(np.uint8)
    n_z = cv2.cvtColor(n_z, cv2.COLOR_GRAY2BGR)

    theta_deg = ((theta * 180 / np.pi) + 90) / 180
    n_xy = (theta_deg * 255).astype(np.uint8)
    n_xy = cv2.applyColorMap(n_xy, cv2.COLORMAP_JET)

    sin_d2 = np.sin(dolp)**2
    sin_t2 = np.sin(theta)**2

    x2 = sin_d2 * sin_t2
    y2 = sin_d2 - x2
    z2 = (1-sin_d2) * dolp_factor**2

    norm = x2+y2+z2
    x2, y2, z2 = [item / norm for item in [x2, y2, z2]]
    x, y, z = [np.sqrt(item) for item in [x2, y2, z2]]

    n_xyz = np.stack([z, y, x], axis=2)
    n_xyz = (n_xyz * 255).astype(np.uint8)

    return n_z, n_xy, n_xyz, u_i


if __name__ == "__main__":
    args = parse_args()
    data_path = Path("pxi_sources/")
    save_path = Path("normals_output")
    save_path.mkdir(exist_ok=True)

    api = vyz.vyzAPI()

    dolp_factor = 0.05
    scale_factor = 1.8

    pxi_files = list(data_path.glob("*.pxi"))

    our_frame_source = FrameSource("pxi_sources/", None, None)
    our_frame_source.file_list_pxi = pxi_files
    our_frame_source_iter = iter(our_frame_source)

    for i, sample_path in tqdm(enumerate(pxi_files), total=len(pxi_files)):
        api.grabRawFrame(str(sample_path))

        api.processForDisplay()

        save_item_path = save_path / f"{sample_path.stem}"
        save_item_path.mkdir(exist_ok=True)

        i0 = np.asarray(api.getData(vyz.vyzDataType.i0), dtype=np.uint8)
        i45 = np.asarray(api.getData(vyz.vyzDataType.i45), dtype=np.uint8)
        i90 = np.asarray(api.getData(vyz.vyzDataType.i90), dtype=np.uint8)
        i135 = np.asarray(api.getData(vyz.vyzDataType.i135), dtype=np.uint8)

        s0 = np.asarray(api.getData(vyz.vyzDataType.s0), dtype=np.uint8)
        s1 = np.asarray(api.getData(vyz.vyzDataType.s1), dtype=np.uint8)
        s2 = np.asarray(api.getData(vyz.vyzDataType.s2), dtype=np.uint8)

        xyz = np.asarray(api.getData(vyz.vyzDataType.xyz), dtype=np.uint8)
        xy = np.asarray(api.getData(vyz.vyzDataType.xy), dtype=np.uint8)
        z = np.asarray(api.getData(vyz.vyzDataType.z), dtype=np.uint8)
        raw_ui = np.asarray(api.getData(vyz.vyzDataType.raw), dtype=np.uint8)

        s0_our = np.float32(i0) + i90
        s1_our = np.float32(i90) - i0
        s2_our = np.float32(i45) - i135

        n_z, n_xy, n_xyz, u_i = get_normals(s0_our, s1_our, s2_our)

        cv2.imwrite(str(save_item_path / "i0.png"), i0)
        cv2.imwrite(str(save_item_path / "i45.png"), i45)
        cv2.imwrite(str(save_item_path / "i90.png"), i90)
        cv2.imwrite(str(save_item_path / "i135.png"), i135)

        cv2.imwrite(str(save_item_path / "s0.png"), s0)
        cv2.imwrite(str(save_item_path / "s1.png"), s1)
        cv2.imwrite(str(save_item_path / "s2.png"), s2)

        cv2.imwrite(str(save_item_path / "s0_python.png"), s0_our)
        cv2.imwrite(str(save_item_path / "s1_python.png"), s1_our)
        cv2.imwrite(str(save_item_path / "s2_python.png"), s2_our)

        cv2.imwrite(str(save_item_path / "n_z.png"), n_z)
        cv2.imwrite(str(save_item_path / "n_xyz.png"), n_xyz)
        cv2.imwrite(str(save_item_path / "n_xy.png"), n_xy)
        cv2.imwrite(str(save_item_path / "u_i.png"), u_i)

        cv2.imwrite(str(save_item_path / "xy.png"), xy)
        cv2.imwrite(str(save_item_path / "xyz.png"), xyz)
        cv2.imwrite(str(save_item_path / "z.png"), z)
        cv2.imwrite(str(save_item_path / "raw.png"), raw_ui)

        img_i, img_nz, img_nxy, img_nxyz = next(our_frame_source_iter)

        cv2.imwrite(str(save_item_path / "current_people-track_u_i.png"), img_i)
        cv2.imwrite(str(save_item_path / "current_people-track_nz.png"), img_nz)
        cv2.imwrite(str(save_item_path / "current_people-track_nxy.png"), img_nxy)
        cv2.imwrite(str(save_item_path / "current_people-track_nxyz.png"), img_nxyz)

        nz_diff = cv2.absdiff(cv2.cvtColor(img_nz, cv2.COLOR_BGR2GRAY), z)
        nxy_diff = cv2.absdiff(img_nxy, xy)
        nxyz_diff = cv2.absdiff(img_nxyz, xyz)
        cv2.imwrite(str(save_item_path / "diff_z.png"), nz_diff)
        cv2.imwrite(str(save_item_path / "diff_xy.png"), nxy_diff)
        cv2.imwrite(str(save_item_path / "diff_xyz.png"), nxyz_diff)
