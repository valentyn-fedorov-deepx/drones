import numpy as np
from argparse import ArgumentParser
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as manimation
from tqdm import tqdm
import cv2
import pickle as pkl
from collections import defaultdict
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
import random

from .gen_video_from_record import create_person_from_logs, plot_one_box, ResultsAnnotator


def stack_images_vertically(image1, image2, resize_method='max', background_color=(255, 255, 255)):
    """
    Stack two images vertically, handling different sizes.
    
    Parameters:
    -----------
    image1 : numpy.ndarray or PIL.Image
        First image (top)
    image2 : numpy.ndarray or PIL.Image
        Second image (bottom)
    resize_method : str, optional
        Method to handle different widths:
        - 'max': Resize smaller image to match larger width
        - 'min': Resize larger image to match smaller width
        - 'stretch': Stretch both images to mean width
        Default is 'max'
    background_color : tuple, optional
        RGB color for padding (default: white)
        
    Returns:
    --------
    numpy.ndarray
        Stacked image array
    """
    # Convert numpy arrays to PIL Images if necessary
    if isinstance(image1, np.ndarray):
        image1 = Image.fromarray(image1.astype('uint8'))
    if isinstance(image2, np.ndarray):
        image2 = Image.fromarray(image2.astype('uint8'))

    # Get dimensions
    w1, h1 = image1.size
    w2, h2 = image2.size

    # Determine target width based on method
    if resize_method == 'max':
        target_width = max(w1, w2)
    elif resize_method == 'min':
        target_width = min(w1, w2)
    elif resize_method == 'stretch':
        target_width = int((w1 + w2) / 2)
    else:
        raise ValueError("resize_method must be 'max', 'min', or 'stretch'")

    # Calculate new heights maintaining aspect ratio
    new_h1 = int(h1 * (target_width / w1))
    new_h2 = int(h2 * (target_width / w2))

    # Resize images
    image1_resized = image1.resize((target_width, new_h1), Image.Resampling.LANCZOS)
    image2_resized = image2.resize((target_width, new_h2), Image.Resampling.LANCZOS)

    # Create new image with combined height
    total_height = new_h1 + new_h2
    stacked_image = Image.new('RGB', (target_width, total_height), background_color)

    # Paste images
    stacked_image.paste(image1_resized, (0, 0))
    stacked_image.paste(image2_resized, (0, new_h1))

    return np.array(stacked_image)


def pad_to_square(image, background_color=(255, 255, 255), return_pil=False):
    """
    Pad an image to make it square while maintaining aspect ratio.

    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Input image
    background_color : tuple, optional
        RGB color for padding (default: white)
    return_pil : bool, optional
        If True, returns PIL Image instead of numpy array (default: False)

    Returns:
    --------
    numpy.ndarray or PIL.Image
        Square padded image
    """
    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    # Get original dimensions
    width, height = image.size

    # Find the larger dimension
    max_dim = max(width, height)

    # Create a new square image with the background color
    padded_image = Image.new('RGB', (max_dim, max_dim), background_color)

    # Calculate padding
    x_pad = (max_dim - width) // 2
    y_pad = (max_dim - height) // 2

    # Paste original image onto padded image
    padded_image.paste(image, (x_pad, y_pad))

    return padded_image if return_pil else np.array(padded_image)


def plot_to_numpy_via_buffer(fig, dpi=100, transparent=False):
    """
    Alternative method using BytesIO buffer.
    This method might be more memory efficient for large plots.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, transparent=transparent)
    buf.seek(0)
    
    # Read the image back into a NumPy array
    from PIL import Image
    img = Image.open(buf)
    X = np.asarray(img)
    
    # If transparency is not needed and present, convert to RGB
    if not transparent and X.shape[2] == 4:
        X = X[:, :, :3]
    
    return X

def get_data(data_path: Path, reference_person_ids: list, meas_algo: str = "sloth"):

    with open(data_path, 'r') as f:
        distances = json.load(f)
    
    meas_data = dict()

    for person_id in reference_person_ids:
        algo_distances = distances[str(person_id)][meas_algo]
        algo_distances = {key: value for key, value in algo_distances.items() if not np.isnan(value)}
        meas_data.update(algo_distances)

    return meas_data


def plot_to_numpy(fig, dpi=100, transparent=False):
    """
    Convert a matplotlib figure to a NumPy array.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to convert
    dpi : int, optional
        Resolution of the output array (default: 100)
    transparent : bool, optional
        Whether to use transparent background (default: False)

    Returns:
    --------
    numpy.ndarray
        RGB(A) array representing the plot
    """
    # Draw the figure onto the canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the RGBA buffer from the canvas
    buf = canvas.buffer_rgba()

    # Convert to a NumPy array
    X = np.asarray(buf)

    # If transparency is not needed, convert to RGB
    if not transparent:
        X = X[:, :, :3]

    return X


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--measurements-path")
    parser.add_argument(
        '--track-ids',
        type=int,
        nargs='+',
        help='List of integers (space-separated). Example: --numbers 1 2 3'
    )
    parser.add_argument('--pkl-path')
    parser.add_argument('--video-path', type=Path)
    parser.add_argument('--save-path', type=Path, default="output/animation")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    meas_algos = [
                #   'window: 5',
                  'window: 10',
                #   'window: 20',
                #   'window: 30',
                #   'window: 40',
                #   'window: 50',
                  'window: 100',
                #   'window: 120',
                  'window: 140',
                #   'window: 160',
                  'window: 1502000']
    

    # meas_algos = [
    #     "original",
    #     "filtering_outliers"
    # ]
    
    change_name = {'window: 5': "5",
                   'window: 10': "10",
                   'window: 20': "20",
                   'window: 30': "30",
                   'window: 40': "40",
                   'window: 50': "50",
                   'window: 100': "100",
                   'window: 120': "120",
                   'window: 140': "140",
                   'window: 1502000': "original"}

        # "ema_0.11+bones_coef_change",
        # "ema_0.09+bones_coef_change",
        # "ema_0.01+bones_coef_change",
        # "original"

        # ]

    meas_data_per_algo = dict()
    sorted_frame_idxs = set()
    
    dist_min = 1e8
    dist_std = 1e8
    dist_max = 0

    for meas_algo in meas_algos:
        meas_data = get_data(args.measurements_path, args.track_ids, meas_algo)
        dist_np = np.array(list(meas_data.values()))

        dist_min = min(dist_min, dist_np.min())
        dist_std = min(dist_std, dist_np.std())
        dist_max = max(dist_max, dist_np.max())

        sorted_frame_idxs.update(sorted(list(map(int, meas_data.keys()))))
        meas_data_per_algo[meas_algo] = meas_data


    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(24, 5))
    ax.yaxis.set_ticks(np.linspace(dist_min, dist_max, 6))

    # import ipdb; ipdb.set_trace()
    # ln, = ax.plot([], [])

    meas_algos_data = dict()
    for meas_algo in meas_algos:
        color = tuple(random.randint(0, 255) / 255 for i in range(3))
        line = Line2D([], [], label=change_name.get(meas_algo, meas_algo), color=color)
        meas_algos_data[meas_algo] = dict(x=list(), y=list(), ln=line)
    
    # meas_algos_data = {meas_algo: dict(x=list(), y=list(), ln=Line2D([], [], label=meas_algo)) for meas_algo in meas_algos}

    # import ipdb; ipdb.set_trace()
    largest_idx = max(sorted_frame_idxs)
    ax.set_xlim(0, largest_idx+(largest_idx*0.1))
    ax.set_ylim(dist_min - dist_std, dist_max + dist_std)

    for meas_algo, meas_algo_info in meas_algos_data.items():
        ax.add_line(meas_algo_info['ln'])

    legend = ax.legend(
        loc='lower right',            # Position of the legend
        # bbox_to_anchor=(1.25, 0.6),    # Place legend outside the plot
        frameon=True,                  # Add a frame around the legend
        fancybox=True,                 # Round corners
        # shadow=True,                 # Add shadow
        ncol=1                         # Number of columns in legend
    )
    for lh in legend.legend_handles: 
        lh.set_alpha(0.5)

    # x = list()
    # y = list()

    def update_figure(frame_idx: int):
        frame_idx_str = str(frame_idx + 1)
        for meas_algo in meas_algos:
            if frame_idx_str not in meas_data_per_algo[meas_algo]:
                continue
            meas_algos_data[meas_algo]["x"].append(frame_idx)
            meas_algos_data[meas_algo]["y"].append(meas_data_per_algo[meas_algo][frame_idx_str])
            meas_algos_data[meas_algo]["ln"].set_data(meas_algos_data[meas_algo]["x"], meas_algos_data[meas_algo]["y"])

        return [meas_algo_values['ln'] for meas_algo, meas_algo_values in meas_algos_data.items()]

    # FFMpegWriter = manimation.writers['ffmpeg']
    # moviewriter = FFMpegWriter(fps=4)
    # with moviewriter.saving(fig, 'myfile.mp4', dpi=100):
    #     for j in tqdm(range(sorted_frame_idxs[-1])):
    #         update_figure(j)
    #         moviewriter.grab_frame()

    with open(args.pkl_path, "rb") as f:
        data = pkl.load(f)

    annotator = ResultsAnnotator()

    video = cv2.VideoCapture(str(args.video_path))
    fps = 12
    save_path = args.save_path / args.video_path.name
    args.save_path.mkdir(parents=True, exist_ok=True)
    final_shape = (2448, 2448)
    video_out = cv2.VideoWriter(str(save_path),
                                cv2.VideoWriter.fourcc(*'mp4v'), fps,
                                final_shape, True)

    frame_idx_to_data = defaultdict(list)
    for single_log in data['data_log']:
        frame_idx_to_data[single_log["pos_frame"]] += single_log['res']

    sorted_frame_ids = sorted(set(frame_idx_to_data.keys()))

    for frame_number in tqdm(sorted_frame_ids):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        res, frame = video.read()

        # if frame_number > 10:
        #     break

        frame_data = frame_idx_to_data[frame_number]
        people = list()

        for item in frame_data:
            if item['id'] not in args.track_ids:
                continue

            person = create_person_from_logs(item)

            meas = item['meas']
            if meas is not None:
                meas = f"{meas['dist']:.2f}"
                dist = meas_data.get(str(frame_number + 1))
                if dist is None:
                    person.meas = None
                else:
                    person.meas['dist'] = dist

            people.append(person)
            plot_one_box(item['bbox'], frame, (212, 212, 212),
                         "", 5)

            if item['has_pose']:
                for point, point_conf in zip(item['pose'], item['pose_conf']):
                    if point_conf < 0.5:
                        continue

                    x_p, y_p = map(int, point)
                    cv2.circle(frame, (x_p, y_p), 4, (255, 0, 0), -1)

        frame = annotator.process(frame, frame_number, people, list(), list(),
                                  frame, frame, False)

        update_figure(frame_number)
        progress_plot = plot_to_numpy(fig)
        # progress_plot = plot_to_numpy_via_buffer(fig)
        merged_frame = stack_images_vertically(frame, progress_plot)
        merged_frame = pad_to_square(merged_frame)

        merged_frame = cv2.resize(merged_frame, final_shape)

        video_out.write(merged_frame)
    
    # import ipdb; ipdb.set_trace()

    video.release()
    video_out.release()
