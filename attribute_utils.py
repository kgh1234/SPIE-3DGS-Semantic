# utils.py
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from plyfile import PlyData


def load_pointcept_data(pointcept_dir):
    """
    Pointcept 데이터를 .npy 파일에서 로드.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
    
    Returns:
        dict: 로드된 데이터 (coord, color, normal, segment20, segment200, instance).
              segment200과 instance는 선택적으로 로드됨.
    """
    data = {}
    # 필수 데이터 로드
    required_keys = ['coord', 'color', 'normal', 'segment20']
    for key in required_keys:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{key}.npy not found in {pointcept_dir}")
        data[key] = np.load(file_path)

    # 음수 값 경고 (필터링 또는 변환하지 않음)
    if np.any(data['segment20'] < 0):
        print(f"Warning: Negative values found in segment20: {data['segment20'][data['segment20'] < 0]}")
        print(f"Keeping negative values as ignore_index (-1) following Pointcept's approach")

    # 선택적 데이터 로드 (segment200, instance)
    optional_keys = ['segment200', 'instance']
    for key in optional_keys:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if os.path.exists(file_path):
            data[key] = np.load(file_path)
            # 음수 값 경고 (필터링 또는 변환하지 않음)
            if np.any(data[key] < 0):
                print(f"Warning: Negative values found in {key}: {data[key][data[key] < 0]}")
                print(f"Keeping negative values as ignore_index (-1) following Pointcept's approach")
        else:
            print(f"Warning: {key}.npy not found in {pointcept_dir}, setting to ignore_index (-1)")
            data[key] = np.full_like(data['segment20'], -1, dtype=np.int64)

    print(f"Loaded Pointcept data from {pointcept_dir}: {data['coord'].shape[0]} points")
    print(f"Segment20 label distribution: min={data['segment20'].min()}, max={data['segment20'].max()}")
    return data

def load_3dgs_data_with_attributes(path_3dgs):
    """
    Load 3DGS data with attributes from a PLY file.
    
    Args:
        path_3dgs (str): Path to the 3DGS PLY file
    
    Returns:
        points_3dgs (np.ndarray): 3DGS points [N, 3]
        features_3dgs (np.ndarray): 3DGS attributes [N, 8] (scale_x, scale_y, scale_z, opacity, rot_w, rot_x, rot_y, rot_z)
    """
    with open(path_3dgs, 'rb') as f:
        ply_data_3dgs = PlyData.read(f)
    vertex_data_3dgs = ply_data_3dgs['vertex']
    
    points_3dgs = np.stack([vertex_data_3dgs['x'], vertex_data_3dgs['y'], vertex_data_3dgs['z']], axis=-1)
    
    # SH 1차 계수 (f_dc_0, f_dc_1, f_dc_2) -> RGB
    colors_3dgs = np.stack([vertex_data_3dgs['f_dc_0'], vertex_data_3dgs['f_dc_1'], vertex_data_3dgs['f_dc_2']], axis=-1)
    # SH 계수는 일반적으로 [-1, 1] 범위로 저장됨 -> [0, 1]로 정규화
    colors_3dgs = (colors_3dgs + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    colors_3dgs = np.clip(colors_3dgs, 0.0, 1.0)  # 범위 제한

    scales_3dgs = np.stack([vertex_data_3dgs['scale_0'], vertex_data_3dgs['scale_1'], vertex_data_3dgs['scale_2']], axis=-1)
    opacity_3dgs = vertex_data_3dgs['opacity']
    rotation_3dgs = np.stack([vertex_data_3dgs['rot_0'], vertex_data_3dgs['rot_1'], vertex_data_3dgs['rot_2'], vertex_data_3dgs['rot_3']], axis=-1)  # Quaternion (w, x, y, z)
    
    features_3dgs = np.hstack((scales_3dgs, opacity_3dgs[:, np.newaxis], rotation_3dgs))
    return points_3dgs, colors_3dgs, features_3dgs

def preprocessing_attr(raw_features_3dgs, normalize_scale=True):
    """
    Preprocess raw 3DGS attributes to make them usable.
    
    Args:
        raw_features_3dgs (np.ndarray): Raw 3DGS attributes [N, 8] (scale_x, scale_y, scale_z, opacity, rot_w, rot_x, rot_y, rot_z)
        normalize_scale (bool): Whether to normalize scale values to [0, 1]. Default is True.
    
    Returns:
        features_3dgs (np.ndarray): Processed 3DGS attributes [N, 8] (scale_x, scale_y, scale_z, opacity, rot_w, rot_x, rot_y, rot_z)
    """
    # 속성 분리
    scale = raw_features_3dgs[:, 0:3]  # [N, 3] (scale_x, scale_y, scale_z)
    opacity = raw_features_3dgs[:, 3:4]  # [N, 1] (opacity)
    rotation = raw_features_3dgs[:, 4:8]  # [N, 4] (rot_w, rot_x, rot_y, rot_z)

    # Scale: 로그 변환 해제
    scale_processed = np.exp(scale)  # log(scale) -> scale
    scale_processed = np.nan_to_num(scale_processed, nan=1e-6, posinf=1e-6, neginf=1e-6)
    scale_processed = np.maximum(scale_processed, 1e-6)  # 양수 보장

    # Scale 정규화 (옵션)
    if normalize_scale:
        scale_min = scale_processed.min()
        scale_max = scale_processed.max()
        if scale_max > scale_min:  # 분모가 0이 되지 않도록
            scale_processed = (scale_processed - scale_min) / (scale_max - scale_min)
        else:
            scale_processed = np.zeros_like(scale_processed)  # 모든 값이 동일하면 0으로 설정

    # Opacity: 시그모이드 변환
    opacity_processed = 1 / (1 + np.exp(-opacity))  # [-∞, ∞] -> [0, 1]
    opacity_processed = np.nan_to_num(opacity_processed, nan=0.0, posinf=1.0, neginf=0.0)
    opacity_processed = np.clip(opacity_processed, 0.0, 1.0)  # [0, 1]로 클리핑

    # Rotation: 쿼터니언 정규화
    rotation_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    rotation_processed = np.where(rotation_norm > 0, rotation / rotation_norm, rotation)

    # 후처리된 속성 결합
    features_3dgs = np.hstack((scale_processed, opacity_processed, rotation_processed))
    return features_3dgs

def quaternion_to_direction(quaternion):
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    # 기본 방향 벡터 (0, 0, 1)에 쿼터니언 회전 적용
    dir_x = 2 * (x * z + w * y)
    dir_y = 2 * (y * z - w * x)
    dir_z = 1 - 2 * (x * x + y * y)
    return np.stack([dir_x, dir_y, dir_z], axis=-1)

def augment_pointcept_with_3dgs_attributes(points_pointcept, points_3dgs, features_3dgs, k_neighbors=5, use_features=('scale',), aggregation='mean'):
    """
    Pointcept 점에 3DGS 속성을 전달 (KNN 사용).
    
    Args:
        points_pointcept (np.ndarray): Pointcept 점 좌표 [M, 3].
        points_3dgs (np.ndarray): 3DGS 점 좌표 [N, 3].
        features_3dgs (np.ndarray): 3DGS 속성 [N, 8] (scale_x, scale_y, scale_z, opacity, rot_w, rot_x, rot_y, rot_z).
        k_neighbors (int): KNN에서 사용할 이웃 점 개수.
        use_features (tuple): 사용할 features ('scale', 'opacity', 'rotation').
        aggregation (str): 속성 집계 방식 ('mean', 'max', 'median').
    
    Returns:
        np.ndarray: Pointcept 점에 전달된 속성 [M, D].
    """
    # 3DGS 속성 분리
    scale_3dgs = features_3dgs[:, 0:3]  # [N, 3] (scale_x, scale_y, scale_z)
    opacity_3dgs = features_3dgs[:, 3:4]  # [N, 1] (opacity)
    rotation_3dgs = features_3dgs[:, 4:8]  # [N, 4] (rot_w, rot_x, rot_y, rot_z)

    # KNN으로 이웃 찾기
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_3dgs)
    distances, indices = nbrs.kneighbors(points_pointcept)

    # 사용할 features 선택
    selected_features = []
    if 'scale' in use_features:
        # Scale 속성
        scale_neighbors = scale_3dgs[indices]  # [M, k_neighbors, 3]
        if aggregation == 'mean':
            scale_augmented = np.mean(scale_neighbors, axis=1)  # [M, 3]
        elif aggregation == 'max':
            scale_augmented = np.max(scale_neighbors, axis=1)  # [M, 3]
        elif aggregation == 'median':
            scale_augmented = np.median(scale_neighbors, axis=1)  # [M, 3]
        selected_features.append(scale_augmented)

    if 'opacity' in use_features:
        # Opacity 속성
        opacity_neighbors = opacity_3dgs[indices]  # [M, k_neighbors, 1]
        if aggregation == 'mean':
            opacity_augmented = np.mean(opacity_neighbors, axis=1)  # [M, 1]
        elif aggregation == 'max':
            opacity_augmented = np.max(opacity_neighbors, axis=1)  # [M, 1]
        elif aggregation == 'median':
            opacity_augmented = np.median(opacity_neighbors, axis=1)  # [M, 1]
        selected_features.append(opacity_augmented)

    if 'rotation' in use_features:
        # Rotation 속성 (쿼터니언 → 방향 벡터)
        rotation_neighbors = rotation_3dgs[indices]  # [M, k_neighbors, 4]
        # 쿼터니언 평균 계산 (단순 평균은 쿼터니언에 적합하지 않으므로 방향 벡터로 변환 후 평균)
        direction_neighbors = np.array([quaternion_to_direction(rotation_neighbors[i]) for i in range(len(points_pointcept))])  # [M, k_neighbors, 3]
        if aggregation == 'mean':
            direction_augmented = np.mean(direction_neighbors, axis=1)  # [M, 3]
        elif aggregation == 'max':
            direction_augmented = np.max(direction_neighbors, axis=1)  # [M, 3]
        elif aggregation == 'median':
            direction_augmented = np.median(direction_neighbors, axis=1)  # [M, 3]
        selected_features.append(direction_augmented)

    # 선택된 features 연결
    if selected_features:
        augmented_features = np.concatenate(selected_features, axis=-1)  # [M, D]
    else:
        augmented_features = np.zeros((len(points_pointcept), 0), dtype=np.float32)

    return augmented_features

# label_mapping.py

def update_3dgs_labels(points_3dgs, points_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6):
    """
    3DGS 점에 Pointcept 점의 레이블을 KNN으로 매핑하고, -1 레이블 처리 및 일관성 체크를 수행.
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표 [N, 3].
        points_pointcept (np.ndarray): Pointcept 점 좌표 [M, 3].
        labels_pointcept (np.ndarray): Pointcept 점 segment20 라벨 [M].
        labels200_pointcept (np.ndarray): Pointcept 점 segment200 라벨 [M].
        instances_pointcept (np.ndarray): Pointcept 점 instance ID [M].
        k_neighbors (int): 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
        ignore_threshold (float): ignore_index(-1) 라벨의 비율이 이 값 이상이면 결과 라벨을 -1로 설정.
    
    Returns:
        tuple: (labels_3dgs, labels200_3dgs, instances_3dgs, mask)
            - labels_3dgs (np.ndarray): 매핑된 segment20 라벨 [N].
            - labels200_3dgs (np.ndarray): 매핑된 segment200 라벨 [N].
            - instances_3dgs (np.ndarray): 매핑된 instance ID [N].
            - mask (np.ndarray): 일관성 체크를 통과한 포인트의 마스크 [N].
    """
    # KNN을 사용하여 가장 가까운 Pointcept 점 찾기
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_pointcept)
    distances, indices = nbrs.kneighbors(points_3dgs)

    # 레이블 초기화
    labels_3dgs = np.full(len(points_3dgs), -1, dtype=np.int64)  # ignore_index로 초기화
    labels200_3dgs = np.full(len(points_3dgs), -1, dtype=np.int64)  # ignore_index로 초기화
    instances_3dgs = np.full(len(points_3dgs), -1, dtype=np.int64)  # ignore_index로 초기화
    mask = np.ones(len(points_3dgs), dtype=bool)  # 레이블 일관성 체크용 마스크

    for i in range(len(points_3dgs)):
        nearest_labels = labels_pointcept[indices[i]]
        nearest_labels200 = labels200_pointcept[indices[i]]
        nearest_instances = instances_pointcept[indices[i]]

        # ignore_index(-1)의 비율 계산
        ignore_ratio_labels = np.mean(nearest_labels == -1)
        ignore_ratio_labels200 = np.mean(nearest_labels200 == -1)
        ignore_ratio_instances = np.mean(nearest_instances == -1)

        # segment20 레이블 처리
        if ignore_ratio_labels >= ignore_threshold:
            labels_3dgs[i] = -1
        else:
            temp_labels = np.where(nearest_labels < 0, 0, nearest_labels)
            label_counts = np.bincount(temp_labels)
            labels_3dgs[i] = np.argmax(label_counts)

        # segment200 레이블 처리
        if ignore_ratio_labels200 >= ignore_threshold:
            labels200_3dgs[i] = -1
        else:
            temp_labels200 = np.where(nearest_labels200 < 0, 0, nearest_labels200)
            label200_counts = np.bincount(temp_labels200)
            labels200_3dgs[i] = np.argmax(label200_counts)

        # instance 처리
        if ignore_ratio_instances >= ignore_threshold:
            instances_3dgs[i] = -1
        else:
            temp_instances = np.where(nearest_instances < 0, 0, nearest_instances)
            instance_counts = np.bincount(temp_instances)
            instances_3dgs[i] = np.argmax(instance_counts)

        # 라벨 일관성 체크
        if use_label_consistency:
            temp_labels = np.where(nearest_labels < 0, 0, nearest_labels)
            label_counts = np.bincount(temp_labels)
            if label_counts.max() < k_neighbors * 0.6:  # 60% 이상 일관성 없으면 제외
                mask[i] = False

    return labels_3dgs, labels200_3dgs, instances_3dgs, mask

