# By Oleksiy Grechnyev, 4/30/24
# Cache obstacles detected by SAM+DINO

import sys
import pathlib
import pickle
import shutil
import typing

# from src.cv_module.obstacles import Obs
from aux_codes.kos_mos.vol import obstacle


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def check_cache(video_path: str) -> typing.Optional[list[obstacle.Obstacle]]:
    """Check if video_path is in cache, return None if not, otherwise load static obstacles"""
    p_cache_dir = pathlib.Path('./output/obstacle_cache')
    if not p_cache_dir.is_dir():
        return None

    # Wrong video path, we don't use cache
    with open(p_cache_dir / 'video_path.txt') as f:
        if f.read() != video_path:
            return None

    # Load obstacles
    obstacles = []
    flist = sorted([p for p in p_cache_dir.iterdir() if p.suffix == '.pkl'])
    for p in flist:
        with open(p, 'rb') as f:
            ob = pickle.load(f)
        obstacles.append(ob)

    return obstacles


########################################################################################################################
def save_obstacles(video_path: str, obstacles: list[obstacle.Obstacle]):
    """Save static obstacles to the cache"""
    p_cache_dir = pathlib.Path('./output/obstacle_cache')

    if p_cache_dir.is_dir():
        # Remove old cache cotent
        shutil.rmtree(p_cache_dir)
    # and create anew
    p_cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache obstacles indexed by video_path
    with open(p_cache_dir / 'video_path.txt', 'w') as f:
        f.write(video_path)

    for i, ob in enumerate(obstacles):
        p = p_cache_dir / f'{i:04d}.pkl'
        with open(p, 'wb') as f:
            pickle.dump(ob, f)


########################################################################################################################
