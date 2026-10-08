import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

# Pruning 함수들
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_3dgs_attributes(vertex_data_3dgs):
    # 속성 추출
    scales = np.stack([vertex_data_3dgs[f'scale_{i}'] for i in range(3)], axis=-1)  # [N, 3] (scale_x, scale_y, scale_z)
    opacity = vertex_data_3dgs['opacity']  # [N,]
    rotation = np.stack([vertex_data_3dgs[f'rot_{i}'] for i in range(4)], axis=-1)  # [N, 4] (rot_w, rot_x, rot_y, rot_z)

    # Scale: 로그 변환 해제
    scales_processed = np.exp(scales)  # log(scale) -> scale
    scales_processed = np.nan_to_num(scales_processed, nan=1e-6, posinf=1e-6, neginf=1e-6)
    scales_processed = np.maximum(scales_processed, 1e-6)  # 양수 보장

    # Opacity: 시그모이드 변환
    opacity_processed = 1 / (1 + np.exp(-opacity))  # [-∞, ∞] -> [0, 1]
    opacity_processed = np.nan_to_num(opacity_processed, nan=0.0, posinf=1.0, neginf=0.0)
    opacity_processed = np.clip(opacity_processed, 0.0, 1.0)  # [0, 1]로 클리핑

    # Rotation: 쿼터니언 정규화
    rotation_norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
    rotation_processed = np.where(rotation_norm > 0, rotation / rotation_norm, rotation)

    # 전처리된 속성 결합
    features_3dgs = np.hstack((scales_processed, opacity_processed[:, np.newaxis], rotation_processed))
    return features_3dgs

def prune_by_pointcept_distance(points_3dgs, points_pointcept, pdistance_max=0.00008):
    """
    Prune 3DGS points based on distance to the nearest Pointcept point.
    
    Args:
        points_3dgs (np.ndarray): 3DGS points [N, 3].
        points_pointcept (np.ndarray): Pointcept points [M, 3].
        pdistance_max (float): Maximum allowed distance to the nearest Pointcept point (squared distance).
    
    Returns:
        np.ndarray: Boolean mask indicating which 3DGS points to keep.
    """
    num_points_before = len(points_3dgs)
    pcd_pointcept = o3d.geometry.PointCloud()
    pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
    tree = o3d.geometry.KDTreeFlann(pcd_pointcept)
    
    # 각 3DGS 포인트에서 가장 가까운 Pointcept 포인트까지의 제곱 거리 계산
    distances = np.array([tree.search_knn_vector_3d(point, 1)[2][0] for point in points_3dgs])
    
    # 제곱 거리 기준으로 Pruning
    mask = distances <= pdistance_max
    
    num_points_pruned = num_points_before - np.sum(mask)
    print(f"Pointcept Distance Pruning: Before {num_points_before} points, Pruned {num_points_pruned} points with squared distance > {pdistance_max:.5f}, After {np.sum(mask)} points")
    return mask

def prune_3dgs(vertex_data_3dgs, points_3dgs, normals_3dgs, points_pointcept, normals_pointcept, prune_methods, prune_params):
    """
    3DGS 점에 대해 다양한 pruning 방법을 적용.
    
    Args:
        vertex_data_3dgs: 3DGS PLY 파일의 vertex 데이터.
        points_3dgs (np.ndarray): 3DGS 점 좌표 (N, 3).
        normals_3dgs (np.ndarray): 3DGS 점 법선 (N, 3).
        points_pointcept (np.ndarray): Pointcept 점 좌표 (M, 3).
        normals_pointcept (np.ndarray): Pointcept 점 법선 (M, 3).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
    
    Returns:
        tuple: Pruning된 (points_3dgs, normals_3dgs, vertex_data_3dgs).
    """
    print(f"Initial 3DGS points: {len(points_3dgs)}")
    
    # 1. 3DGS 속성 전처리
    if vertex_data_3dgs is not None:
        features_3dgs = preprocess_3dgs_attributes(vertex_data_3dgs)
    else:
        features_3dgs = None

    # 마스크 초기화
    mask = np.ones(len(points_3dgs), dtype=bool)
    
    # 2. Pointcept Distance Pruning (최우선 적용)
    if prune_methods.get('pointcept_distance', False):
        pcd_pointcept = o3d.geometry.PointCloud()
        pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
        tree = o3d.geometry.KDTreeFlann(pcd_pointcept)
        distances = np.array([tree.search_knn_vector_3d(point, 1)[2][0] for point in points_3dgs])
        distance_mask = distances <= prune_params['pointcept_max_distance']
        mask = mask & distance_mask
        print(f"Pointcept Distance Pruning: Before {len(points_3dgs)} points, Pruned {np.sum(~distance_mask)} points with distance > {prune_params['pointcept_max_distance']:.5f}, After {np.sum(mask)} points")
    
    # Pointcept Distance Pruning 후 점 업데이트
    points_3dgs = points_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    vertex_data_3dgs = vertex_data_3dgs[mask] if vertex_data_3dgs is not None else None
    features_3dgs = features_3dgs[mask] if features_3dgs is not None else None
    
    # 점이 0개일 경우 조기 종료
    if len(points_3dgs) == 0:
        print("No 3DGS points remaining after Pointcept Distance Pruning. Skipping further pruning.")
        return points_3dgs, normals_3dgs, vertex_data_3dgs
    
    mask = np.ones(len(points_3dgs), dtype=bool)  # 마스크 초기화
    
    # 3. Scale-based Pruning
    if prune_methods.get('scale', False) and features_3dgs is not None:
        scales = features_3dgs[:, 0:3]  # 전처리된 scale 값 사용

        # 상한값/하한값 기반 Pruning
        scale_lower_threshold = prune_params.get('scale_lower_threshold', 0.01)
        scale_upper_threshold = prune_params.get('scale_upper_threshold', 0.9)
        scale_threshold_mask = np.all((scales >= scale_lower_threshold) & (scales <= scale_upper_threshold), axis=1)
        mask = mask & scale_threshold_mask
        print(f"Scale Threshold Pruning: Pruned {np.sum(~scale_threshold_mask)} points with scale outside [{scale_lower_threshold:.4f}, {scale_upper_threshold:.4f}], Remaining {np.sum(mask)} points")

        # 추가적으로 ratio 기반 Pruning
        if prune_methods.get('scale_ratio', 0.0) > 0:
            scale_magnitudes = np.linalg.norm(scales[mask], axis=-1)
            threshold = np.percentile(scale_magnitudes, 100 * (1 - prune_methods['scale_ratio']))
            scale_ratio_mask = scale_magnitudes <= threshold
            temp_mask = np.ones(len(mask), dtype=bool)
            temp_mask[mask] = scale_ratio_mask
            mask = mask & temp_mask
            print(f"Scale Ratio Pruning: Pruned {np.sum(~scale_ratio_mask)} points with scale > {threshold:.4f}, Remaining {np.sum(mask)} points")
    
    # 4. Opacity-based Pruning
    if prune_methods.get('opacity', False) and features_3dgs is not None:
        opacities = features_3dgs[:, 3]  # 전처리된 opacity 값 사용

        # 상한값/하한값 기반 Pruning
        opacity_lower_threshold = prune_params.get('opacity_lower_threshold', 0.01)
        opacity_upper_threshold = prune_params.get('opacity_upper_threshold', 0.9)
        opacity_threshold_mask = (opacities >= opacity_lower_threshold) & (opacities <= opacity_upper_threshold)
        mask = mask & opacity_threshold_mask
        print(f"Opacity Threshold Pruning: Pruned {np.sum(~opacity_threshold_mask)} points with opacity outside [{opacity_lower_threshold:.4f}, {opacity_upper_threshold:.4f}], Remaining {np.sum(mask)} points")

        # 추가적으로 ratio 기반 Pruning
        if prune_methods.get('opacity_ratio', 0.0) > 0:
            threshold = np.percentile(opacities[mask], 100 * prune_methods['opacity_ratio'])
            opacity_ratio_mask = opacities[mask] >= threshold
            temp_mask = np.ones(len(mask), dtype=bool)
            temp_mask[mask] = opacity_ratio_mask
            mask = mask & temp_mask
            print(f"Opacity Ratio Pruning: Pruned {np.sum(~opacity_ratio_mask)} points with opacity < {threshold:.4f}, Remaining {np.sum(mask)} points")
    
    # 5. Density-based Pruning
    if prune_methods.get('density', False):
        pcd_3dgs = o3d.geometry.PointCloud()
        pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
        pcd_3dgs, ind = pcd_3dgs.remove_radius_outlier(
            nb_points=prune_params['density_min_points'],
            radius=prune_params['density_eps']
        )
        density_mask = np.zeros(len(points_3dgs), dtype=bool)
        density_mask[ind] = True
        mask = mask & density_mask
        print(f"Density Pruning: Pruned {np.sum(~density_mask)} points with density < {prune_params['density_min_points']} points within {prune_params['density_eps']:.4f}, Remaining {np.sum(mask)} points")
    
    # 6. Normal-based Pruning (Pointcept와 함께 고려)
    if prune_methods.get('normal', False):
        # 병합된 Point Cloud 생성
        pcd_merged = o3d.geometry.PointCloud()
        pcd_merged.points = o3d.utility.Vector3dVector(np.vstack((points_pointcept, points_3dgs)))
        pcd_merged.normals = o3d.utility.Vector3dVector(np.vstack((normals_pointcept, normals_3dgs)))
        
        # 병합된 Point Cloud에서 법선 추정
        pcd_merged.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(prune_params['normal_k_neighbors'])
        )
        pcd_merged.orient_normals_consistent_tangent_plane(k=prune_params['normal_k_neighbors'])
        normals_estimated = np.asarray(pcd_merged.normals)
        # 3DGS 점에 해당하는 법선 추출
        normals_estimated_3dgs = normals_estimated[len(points_pointcept):]
        
        # 추정된 법선 벡터 크기 확인
        norm_estimated = np.linalg.norm(normals_estimated_3dgs, axis=-1)
        print(f"Estimated normals norm: min={norm_estimated.min():.4f}, max={norm_estimated.max():.4f}, zero_count={np.sum(norm_estimated == 0)}")
        
        # 추정된 법선이 모두 0인 경우, 3DGS 점만으로 다시 추정
        if np.all(norm_estimated == 0):
            print("All estimated normals are zero. Estimating normals for 3DGS points only...")
            pcd_3dgs = o3d.geometry.PointCloud()
            pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
            pcd_3dgs.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(prune_params['normal_k_neighbors'])
            )
            pcd_3dgs.orient_normals_consistent_tangent_plane(k=prune_params['normal_k_neighbors'])
            normals_estimated_3dgs = np.asarray(pcd_3dgs.normals)
            norm_estimated = np.linalg.norm(normals_estimated_3dgs, axis=-1)
            print(f"After re-estimation - Estimated normals norm: min={norm_estimated.min():.4f}, max={norm_estimated.max():.4f}, zero_count={np.sum(norm_estimated == 0)}")
        
        # 법선 벡터 크기 확인 및 필터링
        norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
        print(f"3DGS normals norm (in prune_3dgs): min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
        valid_norm_mask = (norm_3dgs > 0) & (norm_estimated > 0)
        print(f"Points with valid norms: {np.sum(valid_norm_mask)} / {len(normals_3dgs)}")
        
        # 코사인 유사도 계산 (방향 무시)
        cos_sim = np.zeros(len(normals_3dgs))
        if np.sum(valid_norm_mask) > 0:
            cos_sim[valid_norm_mask] = np.abs(np.sum(normals_3dgs[valid_norm_mask] * normals_estimated_3dgs[valid_norm_mask], axis=-1) / (
                np.linalg.norm(normals_3dgs[valid_norm_mask], axis=-1) * np.linalg.norm(normals_estimated_3dgs[valid_norm_mask], axis=-1)
            ))
        # 코사인 유사도 분포 출력
        print(f"Cosine similarity: min={cos_sim.min():.4f}, max={cos_sim.max():.4f}, mean={cos_sim.mean():.4f}")
        print(f"Points with cos_sim >= {prune_params['normal_cos_threshold']:.4f}: {np.sum(cos_sim >= prune_params['normal_cos_threshold'])}")
        
        normal_mask = cos_sim >= prune_params['normal_cos_threshold']
        mask = mask & normal_mask
        print(f"Normal Pruning (with Pointcept): Before {np.sum(mask)} points, Pruned {np.sum(~normal_mask)} points with cosine similarity < {prune_params['normal_cos_threshold']:.4f}, After {np.sum(mask)} points")
    
    # 7. SOR-based Pruning
    if prune_methods.get('sor', False):
        pcd_3dgs = o3d.geometry.PointCloud()
        pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
        pcd_3dgs, ind = pcd_3dgs.remove_statistical_outlier(
            nb_neighbors=prune_params['sor_nb_neighbors'],
            std_ratio=prune_params['sor_std_ratio']
        )
        sor_mask = np.zeros(len(points_3dgs), dtype=bool)
        sor_mask[ind] = True
        mask = mask & sor_mask
        print(f"SOR Pruning: Pruned {np.sum(~sor_mask)} points, Remaining {np.sum(mask)} points")
    
    # 최종 점 업데이트
    points_3dgs = points_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    vertex_data_3dgs = vertex_data_3dgs[mask] if vertex_data_3dgs is not None else None
    
    print(f"Final 3DGS points after pruning: {len(points_3dgs)} (Pruned {len(mask) - np.sum(mask)} points in total)")
    return points_3dgs, normals_3dgs, vertex_data_3dgs