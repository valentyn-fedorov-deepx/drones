import numpy as np
import open3d as o3d
from collections import namedtuple
import random

# Namedtuple for storing plane data
PlaneData = namedtuple("PlaneData", ["plane_id", "plane_color", "plane_model", "point_cloud", "missed_frames"])

def normalize_plane_model(plane_model):
    """
    Normalize plane model [A, B, C, D] so that sqrt(A^2 + B^2 + C^2) = 1.
    Returns (normal, d), where normal = (nx, ny, nz), d is the offset.
    
    If the normal length is near zero, returns (0,0,0),0 to avoid numeric issues.
    """
    A, B, C, D = plane_model
    norm_len = np.sqrt(A*A + B*B + C*C)
    if norm_len < 1e-9:
        return (0.0, 0.0, 0.0), 0.0
    return (A/norm_len, B/norm_len, C/norm_len), D/norm_len

def angle_between_normals(n1, n2):
    """
    Compute the angle between two 3D normals n1, n2 (each a tuple/list of length 3).
    Returns the absolute angle in radians, between 0 and pi.
    """
    n1 = np.array(n1, dtype=float)
    n2 = np.array(n2, dtype=float)
    dot = np.dot(n1, n2)
    dot = np.clip(dot, -1.0, 1.0)  # numerical safety
    return np.arccos(abs(dot))  # abs => handle parallel or antiparallel

class PlaneTracker:
    def __init__(self, angle_threshold_degs=5.0, offset_threshold=0.02, max_missed_frames=5):
        """
        Initialize the PlaneTracker.
        
        :param angle_threshold_degs: Maximum angle difference (in degrees) to consider planes the same.
        :param offset_threshold: Maximum offset difference in plane equation to consider planes the same.
        :param max_missed_frames: Maximum number of consecutive frames a plane can be missed before being removed.
        """
        self.angle_threshold = np.radians(angle_threshold_degs)
        self.offset_threshold = offset_threshold
        self.max_missed_frames = max_missed_frames

        self.tracked_planes = []   # list of PlaneData
        self.next_plane_id = 0

        # Define a color palette or random color generator
        self.color_palette = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (0.8, 0.5, 0.2),
            (0.8, 0.2, 0.5),
        ]
        self.color_index = 0

    def _get_next_plane_color(self):
        """
        Get the next color from the palette, cycling if necessary.
        """
        # color = self.color_palette[self.color_index % len(self.color_palette)]
        color = [random.random() for i in range(3)]
        self.color_index += 1
        return color

    def _match_plane_to_tracked(self, plane_model):
        """
        Attempt to match 'plane_model' ([A,B,C,D]) with one of the existing tracked planes.
        Returns the index in self.tracked_planes if matched, else returns -1.
        """
        # Normalize plane model
        normal_i, offset_i = normalize_plane_model(plane_model)
        if np.linalg.norm(normal_i) < 1e-9:
            # Degenerate plane, skip
            return -1

        for idx, p_data in enumerate(self.tracked_planes):
            tracked_model = p_data.plane_model
            normal_t, offset_t = normalize_plane_model(tracked_model)

            # Check angle
            angle_diff = angle_between_normals(normal_i, normal_t)
            if angle_diff > self.angle_threshold:
                continue  # Not similar enough in orientation

            # Check offset
            dotval = np.dot(normal_i, normal_t)
            if dotval >= 0:
                offset_diff = abs(offset_i - offset_t)
            else:
                offset_diff = abs(offset_i + offset_t)

            if offset_diff <= self.offset_threshold:
                # It's a match
                return idx

        return -1

    def update(self, frame_planes):
        """
        Update the tracker with the detected planes from the current frame.
        
        :param frame_planes: List of (plane_model, inlier_cloud) for the current frame.
                             plane_model = [A,B,C,D]
                             inlier_cloud = o3d.geometry.PointCloud
        """
        # Keep track of which tracked planes have been matched this frame
        matched_tracked_indices = set()

        # To store updates for matched planes
        updated_planes = {}

        for plane_model, inlier_cloud in frame_planes:
            # Try to match with an existing tracked plane
            match_idx = self._match_plane_to_tracked(plane_model)
            if match_idx >= 0:
                # Matched with tracked plane
                tracked_plane = self.tracked_planes[match_idx]
                
                # Update PlaneData: reset missed_frames, update model and point cloud
                updated_plane = PlaneData(
                    plane_id=tracked_plane.plane_id,
                    plane_color=tracked_plane.plane_color,
                    plane_model=plane_model,
                    point_cloud=inlier_cloud,
                    missed_frames=0
                )
                updated_planes[match_idx] = updated_plane
                matched_tracked_indices.add(match_idx)
            else:
                # New plane: assign a new ID and color
                pid = self.next_plane_id
                self.next_plane_id += 1
                color = self._get_next_plane_color()

                new_plane = PlaneData(
                    plane_id=pid,
                    plane_color=color,
                    plane_model=plane_model,
                    point_cloud=inlier_cloud,
                    missed_frames=0
                )
                self.tracked_planes.append(new_plane)
                # No need to add to matched_tracked_indices since it's new

        # Increment missed_frames for unmatched tracked planes
        for idx, p_data in enumerate(self.tracked_planes):
            if idx not in matched_tracked_indices:
                # Increment missed_frames
                updated_plane = PlaneData(
                    plane_id=p_data.plane_id,
                    plane_color=p_data.plane_color,
                    plane_model=p_data.plane_model,
                    point_cloud=p_data.point_cloud,
                    missed_frames=p_data.missed_frames + 1
                )
                updated_planes[idx] = updated_plane

        # Update the tracked_planes list with updated_planes
        for idx, updated_plane in updated_planes.items():
            self.tracked_planes[idx] = updated_plane

        # Remove planes that have missed too many frames
        self.tracked_planes = [
            p_data for p_data in self.tracked_planes
            if p_data.missed_frames <= self.max_missed_frames
        ]

    def get_tracked_planes(self):
        """
        Return the list of currently tracked planes as PlaneData namedtuples.
        """
        return self.tracked_planes

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate sequential frames with plane detections
    # In practice, you'd run sequential_ransac_plane_segmentation on each frame's point cloud

    # Example detected planes for Frame 1
    frame1_planes = [
        ([0.0, 0.0, 1.0, -1.0], o3d.geometry.PointCloud()),  # Plane 1
        ([1.0, 1.0, 0.0, -2.0], o3d.geometry.PointCloud())   # Plane 2
    ]

    # Example detected planes for Frame 2 (Plane 1 slightly shifted, Plane 2 same, new Plane 3)
    frame2_planes = [
        ([0.0, 0.0, 1.0, -0.95], o3d.geometry.PointCloud()),  # Plane 1 (matched)
        ([1.0, 1.0, 0.0, -2.0], o3d.geometry.PointCloud()),   # Plane 2 (matched)
        ([0.0, 1.0, 1.0, -1.5], o3d.geometry.PointCloud())    # Plane 3 (new)
    ]

    # Example detected planes for Frame 3 (Plane 1 missing, Plane 2 slightly shifted, Plane 3 matched)
    frame3_planes = [
        ([1.0, 1.0, 0.0, -2.1], o3d.geometry.PointCloud()),   # Plane 2 (matched with Plane 2)
        ([0.0, 1.0, 1.0, -1.45], o3d.geometry.PointCloud())   # Plane 3 (matched with Plane 3)
        # Plane 1 is missing
    ]

    # Create the tracker
    plane_tracker = PlaneTracker(angle_threshold_degs=5.0, offset_threshold=0.1, max_missed_frames=2)

    # Frame 1 update
    print("=== Frame 1 ===")
    plane_tracker.update(frame1_planes)
    for p in plane_tracker.get_tracked_planes():
        print(p)

    # Frame 2 update
    print("\n=== Frame 2 ===")
    plane_tracker.update(frame2_planes)
    for p in plane_tracker.get_tracked_planes():
        print(p)

    # Frame 3 update
    print("\n=== Frame 3 ===")
    plane_tracker.update(frame3_planes)
    for p in plane_tracker.get_tracked_planes():
        print(p)

    # Frame 4 update (Plane 1 still missing, Plane 2 slightly shifted, Plane 3 missing)
    frame4_planes = [
        ([1.0, 1.0, 0.0, -2.15], o3d.geometry.PointCloud())    # Plane 2 (matched)
        # Plane 3 is missing
    ]
    print("\n=== Frame 4 ===")
    plane_tracker.update(frame4_planes)
    for p in plane_tracker.get_tracked_planes():
        print(p)

    # Frame 5 update (Plane 2 missing)
    frame5_planes = [
        # No planes detected
    ]
    print("\n=== Frame 5 ===")
    plane_tracker.update(frame5_planes)
    for p in plane_tracker.get_tracked_planes():
        print(p)

    # Frame 6 update (Plane 1 reappears)
    frame6_planes = [
        ([0.0, 0.0, 1.0, -1.05], o3d.geometry.PointCloud()),  # Plane 1 (reappears)
    ]
    print("\n=== Frame 6 ===")
    plane_tracker.update(frame6_planes)
    for p in plane_tracker.get_tracked_planes():
        print(p)
