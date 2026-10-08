import numpy as np
import open3d as o3d
from scipy.spatial import distance_matrix

def load_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

def farthest_point_sampling(points, num_samples):
    farthest_pts = [points[np.random.randint(len(points))]]
    distances = distance_matrix(points, np.array(farthest_pts))

    for _ in range(1, num_samples):
        idx = np.argmax(distances.min(axis=1))
        farthest_pts.append(points[idx])
        new_distances = distance_matrix(points, np.array([points[idx]]))
        distances = np.minimum(distances, new_distances)

    return np.array(farthest_pts)

def knn_grouping(all_points, fps_points, k):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    grouped_points = []
    for center in fps_points:
        _, idx, _ = kdtree.search_knn_vector_3d(center, k)
        grouped_points.extend(all_points[idx])

    return np.unique(np.array(grouped_points), axis=0)

def save_ply(points, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)

def process_3dgs_ply(ply_path, output_path, num_fps=1024, knn_k=20):
    print(f"Loading: {ply_path}")
    points = load_ply(ply_path)
    print(f"원래 point 개수: {len(points)}")

    print(f"FPS sampling {num_fps} points...")
    fps_pts = farthest_point_sampling(points, num_fps)

    print(f"KNN grouping with k={knn_k} for each FPS point...")
    final_points = knn_grouping(points, fps_pts, knn_k)

    print(f"샘플링 후 point 개수: {len(final_points)}")

    print(f"Saving to: {output_path}")
    save_ply(final_points, output_path)
    print("Done!")

def voxel_downsample_3dgs(ply_path, output_path, voxel_size=0.02):
    print(f"Loading: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"원래 point 개수: {len(pcd.points)}")

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"voxel_size: {voxel_size} 적용 후 point 개수: {len(down_pcd.points)}")

    print(f"Saving to: {output_path}")
    o3d.io.write_point_cloud(output_path, down_pcd)
    print("Done!")

ply_path = "./data/scannet-3DGS/scene0011_00/point_cloud.ply"
output_path = "./outputs/sampling_test/3dgs_scene0011_00-sampling-v004.ply"

# process_3dgs_ply(
#     ply_path=ply_path,
#     output_path=output_path,
#     num_fps=2048,  # FPS point 수
#     knn_k=10       # 각 FPS 주변 KNN 개수
# )

voxel_downsample_3dgs(
    ply_path=ply_path,
    output_path=output_path,
    voxel_size=0.04  # 실험적으로 0.01 ~ 0.04 권장
)