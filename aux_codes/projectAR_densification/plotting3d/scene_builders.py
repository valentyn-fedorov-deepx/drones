import open3d as o3d
from plotly import graph_objects as go
import numpy as np
from processing3d.pcd_manipulation import subsample_ply_points_colors

def get_axes_arrows_for_plotting(centers: np.ndarray, axes: np.ndarray, axis_length: float=0.5):
    x_arrow_x, x_arrow_y, x_arrow_z = [], [], []
    y_arrow_x, y_arrow_y, y_arrow_z = [], [], []
    z_arrow_x, z_arrow_y, z_arrow_z = [], [], []
    
    # Loop over each camera and compute arrow endpoints.
    for i in range(centers.shape[0]):
        cx, cy, cz = centers[i]
        
        # X axis arrow
        x_dir = axes[i, 0, :]
        x_end = centers[i] + axis_length * x_dir
        x_arrow_x.extend([cx, x_end[0], None])
        x_arrow_y.extend([cy, x_end[1], None])
        x_arrow_z.extend([cz, x_end[2], None])
        
        # Y axis arrow
        y_dir = axes[i, 1, :]
        y_end = centers[i] + axis_length * y_dir
        y_arrow_x.extend([cx, y_end[0], None])
        y_arrow_y.extend([cy, y_end[1], None])
        y_arrow_z.extend([cz, y_end[2], None])
        
        # Z axis arrow
        z_dir = axes[i, 2, :]
        z_end = centers[i] + axis_length * z_dir
        z_arrow_x.extend([cx, z_end[0], None])
        z_arrow_y.extend([cy, z_end[1], None])
        z_arrow_z.extend([cz, z_end[2], None])
    return (
        x_arrow_x, x_arrow_y, x_arrow_z,
        y_arrow_x, y_arrow_y, y_arrow_z,
        z_arrow_x, z_arrow_y, z_arrow_z
    )
    


def get_3D(data, width: int = 1224, height: int = 1024, camera: dict=None) -> go.Figure:
    """
    Function to get 3D representation of the data
    :param data: data to be represented
    :return: 3D representation of the data
    """
    fig = go.Figure(data=data)

    # set aspectmode data
    fig = fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(
                showticklabels=False,
                showgrid = False
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False
            ),
            zaxis=dict(
                showticklabels=False,
                showgrid=False,
            ),
            camera=camera,
        ),
        template='plotly_white',
        width=width, height=height,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def get_3D_camview(
    points: np.ndarray,
    colors: np.ndarray, 
    camera_center: np.ndarray, 
    camera_axes: np.ndarray,
    n_points: int = 50_000,
    width: int = 1224, height: int = 1024,
    downscale_factor: int = 3,
    zoom=1.0,
    marker_size: int = 2,
    ):
    """
    Function to get 3D representation of the data
    :param points: data to be represented
    :param colors: colors of the data
    :param camera_center: camera center
    :param camera_axes: camera axes
    :param n_points: number of points to be represented
    :param zoom: zoom factor
    :return: 3D representation of the data
    """
    # normalize scene to [-1..1] cube
    mins, maxs = points.min(axis=0), points.max(axis=0)
    mid = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2
    points_dom = (points - mid) / scale
    # subsample
    n_points = min(n_points, points_dom.shape[0])
    sub_points, sub_colors = subsample_ply_points_colors(points_dom, colors, n_points)
    # build camera list
    eye_w = camera_center
    x_ax, y_ax, z_ax = camera_axes
    forward = z_ax / np.linalg.norm(z_ax)
    up = -y_ax / np.linalg.norm(y_ax)
    # normalize into domain
    eye_dom = (eye_w - mid) / scale
    # simulate camera zoom with zoom variable
    eye_dom = eye_dom / zoom
    center_dom = eye_dom + forward
    camera = {
        "eye":    {"x": float(eye_dom[0]),    "y": float(eye_dom[1]),    "z": float(eye_dom[2])},
        "center": {"x": float(center_dom[0]), "y": float(center_dom[1]), "z": float(center_dom[2])},
        "up":     {"x": float(up[0]),         "y": float(up[1]),         "z": float(up[2])},
    }
    # main point cloud scatter
    scatter = go.Scatter3d(
        x=sub_points[:,0], y=sub_points[:,1], z=sub_points[:,2],
        mode='markers',
        marker=dict(size=marker_size, color=sub_colors / 255),
        name='points'
    )
    
    fig = go.Figure(
        data=[scatter],
        layout=go.Layout(
            showlegend=False,
            scene=dict(
                aspectmode='data',
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                zaxis=dict(showticklabels=False, showgrid=False),
                camera=camera,
            ),
            width=width//downscale_factor, height=height//downscale_factor,
            margin=dict(l=0, r=0, t=10, b=10),
            template='plotly_white',
        )
    )
    return fig
    


def get_3D_camera_animation(
    points: np.ndarray,
    colors: np.ndarray,
    camera_centers: np.ndarray,
    camera_axes: np.ndarray,
    n_points: int=50_000,
    frame_duration_ms=33,
    width=1224, height=1024,
    zoom = 1.0,
    downscale_factor: int=3,
    marker_size: int = 2
    ):
    
    mins, maxs = points.min(axis=0), points.max(axis=0)
    mid = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2
    
    points_dom = (points-mid) / scale
    
    n_points = min(n_points, points_dom.shape[0])
    sub_points, sub_colors = subsample_ply_points_colors(points_dom, colors, n_points)
    
    camera_list = []
    for camera_center, axes in zip(camera_centers, camera_axes):
        eye_w = camera_center
        x_ax, y_ax, z_ax = axes
        forward = z_ax / np.linalg.norm(z_ax)
        up = -y_ax / np.linalg.norm(y_ax)
        
        # normalize into domain
        eye_dom = (eye_w - mid) / scale
        # simulate camera zoom with zoom variable
        eye_dom = eye_dom / zoom
        

        center_dom = eye_dom + forward
        
        camera_list.append({
        "eye":    {"x": float(eye_dom[0]),    "y": float(eye_dom[1]),    "z": float(eye_dom[2])},
        "center": {"x": float(center_dom[0]), "y": float(center_dom[1]), "z": float(center_dom[2])},
        "up":     {"x": float(up[0]),         "y": float(up[1]),         "z": float(up[2])},
    })
        
    scatter = go.Scatter3d(
        x=sub_points[:,0],
        y=sub_points[:,1],
        z=sub_points[:,2],
        mode='markers',
        marker=dict(size=marker_size, color=sub_colors / 255),
    )
    
    frames = [
        go.Frame(
            layout=go.Layout(scene_camera=cam),
            name=str(i)
        )
        for i, cam in enumerate(camera_list)
    ]
    
    fig = go.Figure(
        data=[scatter],
        frames=frames,
        layout=go.Layout(
            scene=dict(
                aspectmode='data',
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                zaxis=dict(showticklabels=False, showgrid=False),
                camera=camera_list[0],
            ),
            width=width//downscale_factor, height=height//downscale_factor,
            margin=dict(l=0, r=0, t=10, b=10),
            updatemenus=[dict(
                type='buttons',
                showactive=False, y=1, x=0.8,
                xanchor='left', yanchor='bottom',
                pad=dict(t=10, r=10),
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=frame_duration_ms, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True, mode='immediate'
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]
                    )
                ]
            )],
            sliders=[dict(
                steps=[
                    dict(
                        method='animate',
                        args=[[str(i)], dict(mode='immediate', frame=dict(duration=0), transition=dict(duration=0))],
                        label=str(i)
                    ) for i in range(len(frames))
                ],
                active=0,
                x=0, y=0,
                currentvalue=dict(prefix='Frame: ', visible=True),
                pad=dict(t=5)
            )],
            template='plotly_white',
        )
    )
        
    return fig

def convert_coords_to_dom(
    target_coords: np.ndarray,
    points: np.ndarray) -> np.ndarray:
    """
    Convert target coordinates to domain coordinates.
    
    :param target_coords: target coordinates to be converted
    :param points: points to be used for normalization
    
    :return: target coordinates in domain
    """
    mins, maxs = points.min(axis=0), points.max(axis=0)
    mid = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2
    target_coords_dom = (target_coords - mid) / scale
    return target_coords_dom
    

def get_3D_camera_animation_planes_track(
    points: np.ndarray,
    colors: np.ndarray,
    camera_centers: np.ndarray,
    camera_axes: np.ndarray,
    plane_centers: np.ndarray,
    plane_axes: np.ndarray,
    n_points: int=50_000,
    points_opacity: float=0.2,
    frame_duration_ms: int=33,
    width: int=1224,
    height: int=1024,
    zoom: float=1.0,
    downscale_factor: int=3,
    axis_length: float=0.1,    # length of each axis arrow in normalized units
    line_width: int=4,        # width of each axis arrow in pixels,
    marker_size: int=2,      # size of each point in pixels
    lift_normal_axis: float = 0, # lift normal axis for camera
    
):
    # normalize scene to [-1..1] cube
    mins, maxs = points.min(axis=0), points.max(axis=0)
    mid = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2
    points_dom = (points - mid) / scale

    # subsample
    n_pts = min(n_points, points_dom.shape[0])
    sub_points, sub_colors = subsample_ply_points_colors(points_dom, colors, n_pts)

    # build camera list
    camera_list = []
    for cam_ctr, axes in zip(camera_centers, camera_axes):
        eye_w = cam_ctr
        x_ax, y_ax, z_ax = axes
        forward = z_ax / np.linalg.norm(z_ax)
        up = -y_ax / np.linalg.norm(y_ax)

        eye_dom = (eye_w - mid) / scale
        eye_dom = eye_dom / zoom
        center_dom = eye_dom + forward

        camera_list.append({
            "eye":    {"x": float(eye_dom[0]),    "y": float(eye_dom[1]),    "z": float(eye_dom[2])},
            "center": {"x": float(center_dom[0]), "y": float(center_dom[1]), "z": float(center_dom[2])},
            "up":     {"x": float(up[0]),         "y": float(up[1]),         "z": float(up[2])},
        })

    # main point cloud scatter
    scatter = go.Scatter3d(
        x=sub_points[:,0], y=sub_points[:,1], z=sub_points[:,2],
        mode='markers',
        marker=dict(size=marker_size, color=sub_colors / 255),
        opacity=points_opacity,
        name='points'
    )

    # build static axis‐triads at each plane center
    axis_traces = []
    for pc, axes in zip(plane_centers, plane_axes):
        # normalize plane center
        pc_dom = (pc - mid) / scale
        # lift the center of the plance in direction of the normal axis
        nz_dir = axes[2] / np.linalg.norm(axes[2])
        pc_dom += nz_dir * lift_normal_axis
        for vec, col, label in zip(axes, ['red','green','blue'], ['X','Y','Z']):
            vec = vec / np.linalg.norm(vec)
            end = pc_dom + vec * axis_length
                
            axis_traces.append(
                go.Scatter3d(
                    x=[pc_dom[0], end[0]],
                    y=[pc_dom[1], end[1]],
                    z=[pc_dom[2], end[2]],
                    mode='lines+markers',
                    marker=dict(size=2, color=col),
                    line=dict(width=line_width, color=col),
                    name=f'plane‐axis {label}'
                )
            )

    # frames only animate camera
    frames = [
        go.Frame(layout=go.Layout(scene_camera=cam), name=str(i))
        for i, cam in enumerate(camera_list)
    ]

    fig = go.Figure(
        data=[scatter] + axis_traces,
        frames=frames,
        layout=go.Layout(
            showlegend=False,
            scene=dict(
                aspectmode='data',
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                zaxis=dict(showticklabels=False, showgrid=False),
                camera=camera_list[0],
            ),
            width=width//downscale_factor, height=height//downscale_factor,
            margin=dict(l=0, r=0, t=10, b=10),
            template='plotly_white',
            updatemenus=[dict(
                type='buttons', showactive=False, y=1, x=0.8,
                xanchor='left', yanchor='bottom',
                pad=dict(t=10, r=10),
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True),
                                          transition=dict(duration=0),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                ]
            )],
            sliders=[dict(
                steps=[dict(method='animate',
                            args=[[str(i)], dict(mode='immediate', frame=dict(duration=0), transition=dict(duration=0))],
                            label=str(i))
                       for i in range(len(frames))],
                active=0, x=0, y=0,
                currentvalue=dict(prefix='Frame: ', visible=True),
                pad=dict(t=5)
            )]
        )
    )

    return fig


def plot_o3d_mesh_in_plotly(mesh: o3d.geometry.TriangleMesh,
                            show_color: bool = True,
                            mesh_opacity: float = 1.0):
    """
    Render an Open3D TriangleMesh in Plotly.
    
    :param mesh:         open3d.geometry.TriangleMesh
    :param show_color:   if True and mesh has vertex_colors, color faces accordingly
    :param mesh_opacity: opacity of the mesh
    """
    # 1) Extract vertices and faces
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    x, y, z = verts.T
    i, j, k = faces.T

    # 2) Build face colors if available
    if show_color and mesh.has_vertex_colors():
        # average the three vertex colors for each face
        vcols = (np.asarray(mesh.vertex_colors) * 255).astype(int)
        facecols = vcols[faces].mean(axis=1).astype(int)
        # turn into CSS-style rgb strings
        face_colors = [
            f"rgb({r},{g},{b})"
            for r, g, b in facecols
        ]
    else:
        face_colors = "lightgray"

    # 3) Create the Plotly Mesh3d trace
    mesh3d = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        facecolor=face_colors,
        opacity=mesh_opacity,
        flatshading=True
    )

    # 4) Build & show figure
    fig = go.Figure(mesh3d)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig