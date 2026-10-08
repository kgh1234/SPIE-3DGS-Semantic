import numpy as np
import open3d as o3d
from plyfile import PlyData
import os
from sklearn.neighbors import NearestNeighbors

def load_pointcept_data(pointcept_dir):
    data = {}
    required_keys = ['coord', 'color', 'normal', 'segment20']
    for key in required_keys:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{key}.npy not found in {pointcept_dir}")
        data[key] = np.load(file_path)

    optional_keys = ['segment200', 'instance']
    for key in optional_keys:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if os.path.exists(file_path):
            data[key] = np.load(file_path)
        else:
            data[key] = np.full_like(data['segment20'], -1, dtype=np.int64)

    print(f"Loaded Pointcept data from {pointcept_dir}: {data['coord'].shape[0]} points")
    print(f"Segment20 label distribution: min={data['segment20'].min()}, max={data['segment20'].max()}")
    return data

def load_3dgs_data(path_3dgs):
    with open(path_3dgs, 'rb') as f:
        ply_data = PlyData.read(f)
    vertex_data = ply_data['vertex']

    points = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    normals = np.stack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']], axis=-1)
    raw_features = np.hstack([
        np.vstack([vertex_data[f'scale_{i}'] for i in range(3)]).T,
        vertex_data['opacity'][:, None],
        np.vstack([vertex_data[f'rot_{i}'] for i in range(4)]).T
    ])

    norm = np.linalg.norm(normals, axis=-1)
    if np.all(norm == 0):
        print("All normals are zero. Estimating...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normals = np.asarray(pcd.normals)
    print(f"Loaded 3DGS data from {path_3dgs}: {points.shape[0]} points")
    return points, normals, raw_features

def voxelize_3dgs(points, features, voxel_size=0.02, k_neighbors=None, k_neighbors_max=15):
    print(f"Voxelizing with voxel_size={voxel_size}...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    voxel_points = np.asarray(down.points, dtype=np.float32)
    print(f"Voxelization complete: Before {len(points)} → After {len(voxel_points)}")
    return voxel_points, None
def average_quaternions(quaternions):
    """
    Quaternion 배열의 평균 계산.
    Args:
        quaternions: (N, 4) 배열
    Returns:
        avg_quaternion: [w, x, y, z]
    """
    avg = np.mean(quaternions, axis=0)
    norm = np.linalg.norm(avg)
    return avg / (norm + 1e-8)

def quaternion_to_direction(quaternion):
    # 쿼터니언 정규화
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = np.divide(quaternion, norm, where=norm != 0, out=np.zeros_like(quaternion))
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # 기준 벡터 [0, 0, 1]에 회전 적용
    directions = np.zeros((len(quaternion), 3))
    directions[:, 0] = 2 * (x * z + w * y)
    directions[:, 1] = 2 * (y * z - w * x)
    directions[:, 2] = 1 - 2 * (x * x + y * y)

    # 방향 벡터 정규화
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.divide(directions, norm, where=norm != 0, out=np.zeros_like(directions))
    
    return directions

def process_rotation(rotation_values):
    """
    Rotation을 Quaternion 평균화 후 Direction 벡터로 변환하고 정규화.
    
    Args:
        rotation_values (np.ndarray): Quaternion 배열 [N, 4] (w, x, y, z).
    
    Returns:
        np.ndarray: 정규화된 Direction 벡터 [3].
    """
    avg_quaternion = average_quaternions(rotation_values)
    direction = quaternion_to_direction(avg_quaternion[np.newaxis, :])[0]
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else np.array([0, 0, 1], dtype=np.float32)
def update_3dgs_attributes(points_3dgs, points_pointcept, colors_pointcept, normals_pointcept,
                           labels_pointcept, labels200_pointcept, instances_pointcept,
                           k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6):
    """
    CPU 기반 3DGS 속성 업데이트 함수. NearestNeighbors로 KNN 검색.
    """
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto')
    nn.fit(points_pointcept)
    dists, indices = nn.kneighbors(points_3dgs)

    def vote(values, ignore_val=-1):
        output = []
        for neighbor_ids in values:
            counts = {}
            for v in neighbor_ids:
                if v == ignore_val:
                    continue
                counts[v] = counts.get(v, 0) + 1
            if len(counts) == 0:
                output.append(ignore_val)
                continue
            majority = max(counts, key=counts.get)
            if counts[majority] / k_neighbors >= ignore_threshold:
                output.append(majority)
            else:
                output.append(ignore_val)
        return np.array(output)

    neighbor_colors = colors_pointcept[indices]
    neighbor_normals = normals_pointcept[indices]
    neighbor_labels = labels_pointcept[indices]
    neighbor_labels200 = labels200_pointcept[indices]
    neighbor_instances = instances_pointcept[indices]

    colors_3dgs = np.mean(neighbor_colors, axis=1).astype(np.uint8)
    normals_3dgs = np.mean(neighbor_normals, axis=1)
    normals_3dgs /= (np.linalg.norm(normals_3dgs, axis=1, keepdims=True) + 1e-8)

    labels_3dgs = vote(neighbor_labels)
    labels200_3dgs = vote(neighbor_labels200)
    instances_3dgs = vote(neighbor_instances)

    if use_label_consistency:
        mask = np.array([
            np.sum(neighbor_labels[i] == labels_3dgs[i]) / k_neighbors >= ignore_threshold
            if labels_3dgs[i] != -1 else False
            for i in range(len(points_3dgs))
        ])
    else:
        mask = np.ones(len(points_3dgs), dtype=bool)

    return colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask