from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
import trimesh
from pygltflib import GLTF2
import numpy as np

def parse_ply_GF(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    points = np.array([[x, y, z] for x, y, z in zip(vertex['x'], vertex['y'], vertex['z'])])
        
    try:
        colors = np.array([[r, g, b] for r, g, b in zip(vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2'])])
    except:
        colors = np.array([[r, g, b] for r, g, b in zip(vertex['red'], vertex['green'], vertex['blue'])])
    
    # min-max normalization on colors
    colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0))
    
    # turn colors into 255 scale
    colors = (colors * 255).astype(np.uint8)
    
    return points, colors

# subsample the points
def subsample_ply_points_colors(points, colors, n_points):
    assert points.shape[0] == colors.shape[0]
    subsample_len = min(n_points, points.shape[0])
    # generate indices randomly
    idx = np.random.choice(points.shape[0], subsample_len, replace=False)
    return points[idx], colors[idx]

def save_ply(points, colors, filename):
    """
    Save points and colors to a .ply file.
    """
    # Create a PLY file with the points and colors
    vertex = np.array(
        list(zip(points[:, 0], points[:, 1], points[:, 2], colors[:, 0], colors[:, 1], colors[:, 2])),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    ply_data = PlyData([PlyElement.describe(vertex, 'vertex')])
    ply_data.write(filename)
    
    
def ply_to_glb(ply_path: str,
               glb_path: str,
               poisson_depth: int = 12,
               scale: float = 1.3,
               flip_axes: tuple = ('z',)):
    # 1. Load & (optionally) flip
    pcd = o3d.io.read_point_cloud(ply_path)
    M = np.eye(4)
    for ax in flip_axes:
        if ax.lower() in 'xyz':
            M['xyz'.index(ax.lower()), 'xyz'.index(ax.lower())] = -1
    pcd.transform(M)

    # 2. Normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=50)

    # 3. Poisson with extra margin
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=scale
    )

    # 5. Transfer colors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    colors = []
    for v in np.asarray(mesh.vertices):
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
        colors.append(pcd.colors[idx[0]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors))

    # 6. Normals on mesh
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # 7. Export GLB
    tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8),
        process=False
    )
    tri.export(glb_path)
    print(f"→ Wrote {len(tri.vertices)} verts, {len(tri.faces)} faces → {glb_path}")

    # 8. Make material double-sided so you can look at your flipped side
    gltf = GLTF2().load(glb_path)
    for mat in gltf.materials:
        mat.doubleSided = True
    gltf.save(glb_path)
    print("→ Materials set to double-sided (no backface culling).")
    
def poisson_reconstruct(points,
                        colors,
                        depth: int = 9,
                        scale: float = 1.1,
                        linear_fit: bool = False,
                        min_triangle_area: float = 0.0):
        """
        Load a point cloud from PLY, estimate normals, and run Poisson reconstruction.
        Optionally filter out small triangles.
        """

        # build Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float)/255.0)

        # estimate normals (required for Poisson)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01, max_nn=30))
        pcd.normalize_normals()

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            scale=scale,
            linear_fit=linear_fit
        )

        # optionally remove low-density or small triangles
        if min_triangle_area > 0:
            # remove triangles by area
            triangles = np.asarray(mesh.triangles)
            verts = np.asarray(mesh.vertices)
            tri_areas = np.linalg.norm(
                np.cross(
                    verts[triangles[:,1]] - verts[triangles[:,0]],
                    verts[triangles[:,2]] - verts[triangles[:,0]]
                ), axis=1) * 0.5
            keep = tri_areas >= min_triangle_area
            mesh = mesh.select_by_index(np.where(keep)[0], triangle=True)

        return mesh
    
def simplify_mesh_quadric(mesh: o3d.geometry.TriangleMesh,
                          target_triangles: int = None,
                          reduction_ratio: float = 0.1) -> o3d.geometry.TriangleMesh:
    """
    Simplify `mesh` by quadric error decimation.
    You can either specify `target_triangles`, or
    set `reduction_ratio` in (0,1), e.g. 0.1 keeps 10% of faces.
    """

    num_tri = np.asarray(mesh.triangles).shape[0]
    if target_triangles is None:
        target_triangles = max(1, int(num_tri * reduction_ratio))

    return mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
