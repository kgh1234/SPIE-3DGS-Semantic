import os, json
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data, voxelize_3dgs, update_3dgs_attributes, fps_knn_sampling, save_ply
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes, remove_duplicates, pdistance_pruning, pruning_3dgs_attr, select_3dgs_features

def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, exp, prune_methods=None, prune_params=None, k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
     # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    colors_pointcept = pointcept_data['color']
    normals_pointcept = pointcept_data['normal']
    labels_pointcept = pointcept_data['segment20']
    labels200_pointcept = pointcept_data['segment200']
    instances_pointcept = pointcept_data['instance']
    
    # 2. 3DGS 데이터 로드
    points_3dgs, normals_3dgs, raw_features_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)

    # 4. 3DGS pdistance Pruning
    points_3dgs, features_3dgs = pdistance_pruning(
        points_3dgs, features_3dgs, points_pointcept, prune_params
    )

    # 6. 3DGS-attr transfer
    print("Augmenting Pointcept points with 3DGS attributes...")
    features_pointcept  = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors
    )
    
    # 5. 3DGS 데이터 Voxelization (옵션)
    voxelize = voxel_size != 0  # voxel_size가 0이 아니면 voxelize 활성화
    if voxelize:
        print("Applying Voxelization to 3DGS points...")
        points_3dgs, features_3dgs = voxelize_3dgs(
            points_3dgs, features_3dgs, voxel_size=voxel_size, k_neighbors_max=20)
    else:
        print("Applying FPS-kNN to 3DGS points...")
        points_3dgs, features_3dgs = fps_knn_sampling(
            points_3dgs, features_3dgs, sample_ratio=0.05, aggregation_method='mean')
    
    # 7. 3DGS-attr Pruning
    points_3dgs, features_3dgs = pruning_3dgs_attr(
        points_3dgs, features_3dgs, prune_methods, prune_params
    )

    features_pointcept, features_3dgs = select_3dgs_features(
        features_pointcept, features_3dgs, use_features=use_features
    )

    # 8. Point cloud color, normals, labels transfer 
    print("Updating 3DGS attributes from Pointcept...")
    colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask = update_3dgs_attributes(
        points_3dgs, points_pointcept, colors_pointcept, normals_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept,
        k_neighbors=k_neighbors, use_label_consistency=True, ignore_threshold=ignore_threshold
    )

    # 라벨 일관성 필터링 적용
    points_3dgs = points_3dgs[mask]
    colors_3dgs = colors_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    labels_3dgs = labels_3dgs[mask]
    labels200_3dgs = labels200_3dgs[mask]
    instances_3dgs = instances_3dgs[mask]
    features_3dgs = features_3dgs[mask]  # 3DGS 속성도 필터링하여 동기화

    # 6.5. -1 라벨 3DGS 점 비율 출력
    ignore_count = np.sum(labels_3dgs == -1)
    total_count = len(labels_3dgs)
    ignore_ratio = ignore_count / total_count if total_count > 0 else 0
    print(f"3DGS points with label -1: {ignore_count}/{total_count} (Ratio: {ignore_ratio:.4f})")

    # 7. 중복 점 제거 (3DGS 점 제거)
    print("Removing duplicate points...")
    points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs = remove_duplicates(
        points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs
    )

    # 8. 병합
    print("Merging Pointcept and 3DGS points...")
    points_merged = np.vstack((points_pointcept, points_3dgs))
    colors_merged = np.vstack((colors_pointcept, colors_3dgs))
    labels_merged = np.concatenate((labels_pointcept, labels_3dgs))
    print(f"Final merged points: {len(points_merged)} (Pointcept: {len(points_pointcept)}, 3DGS: {len(points_3dgs)})")

    # 11. PLY 파일로 저장
    output_dir_path = os.path.join(output_dir, exp)
    os.makedirs(output_dir_path, exist_ok=True)
    output_ply_path = os.path.join(output_dir_path, f"{exp}.ply")
    save_ply(
        points_merged, colors_merged, labels_merged, output_ply_path,
        points_pointcept=points_pointcept, colors_pointcept=colors_pointcept,
        points_3dgs=points_3dgs, colors_3dgs=colors_3dgs
    )


def process_single_scene(scene, input_root, output_root, exp, path_3dgs_root, prune_methods, prune_params, k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
    try:
        # 입력 경로
        pointcept_dir = os.path.join(input_root, "val", scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        # 출력 경로
        output_dir = os.path.join(output_root)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}")
            return

        # 병합 및 PLY 파일로 저장
        merge_pointcept_with_3dgs(
            pointcept_dir, path_3dgs, output_dir, exp, prune_methods, prune_params,
            k_neighbors=k_neighbors, ignore_threshold=ignore_threshold, voxel_size=voxel_size, use_features=use_features
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single ScanNet scene with 3DGS merging and save as PLY for visualization.")
    parser.add_argument(
        "--scene",
        default="scene0011_00",
        help="Scene to process (e.g., scene0011_00)",
    )
    parser.add_argument(
        "--output_root",
        default="./outputs/test_samples10",
        help="Output path for processed data (default: ./outputs/test_samples10)",
    )
    parser.add_argument(
        "--exp",
        default="pdistance00005_vox002",
        help="Experiment name",
    )
    parser.add_argument(
        "--pruning_ratio",
        default=0.0,
        type=float,
        help="Final ratio of points to prune based on importance scores (bottom X%)",
    )
    parser.add_argument(
        "--attr_pruning_ratio",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Pruning ratios for scale, opacity, rotation (default: 0.0 0.0 0.0). Set ratio > 0 to enable pruning for that attribute.",
    )
    parser.add_argument(
        "--pdistance",
        default=0.001,
        type=float,
        help="Pointcept distance threshold. If specified, pointcept distance pruning is enabled.",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.02,
        type=float,
        help="Voxel size for 3DGS voxelization (default: 0.02m). Set to 0 to disable voxelization.",
    )
    parser.add_argument(
        "--use_features",
        nargs='+',
        default=['scale', 'opacity', 'rotation'],
        help="3DGS features to transfer (default: scale opacity rotation)",
    )

    args = parser.parse_args()

    # Config 파일 로드
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_root = config['input_root']
    path_3dgs_root = config['path_3dgs_root']
    meta_root = config['meta_root']
    prune_params = config['prune_params']
    k_neighbors = prune_params['k_neighbors']
    prune_params['pointcept_max_distance'] = args.pdistance
    # Prune methods 설정
    scale_ratio, opacity_ratio, rotation_ratio = args.attr_pruning_ratio

    prune_methods = {
        'pointcept_distance': args.pdistance > 0,
        'scale': scale_ratio > 0,
        'scale_ratio': scale_ratio,
        'opacity': opacity_ratio > 0,
        'opacity_ratio': opacity_ratio,
        'rotation': rotation_ratio > 0,
        'rotation_ratio': rotation_ratio,
    }

    # 단일 Scene 처리
    process_single_scene(
        args.scene,
        input_root,
        args.output_root,
        args.exp,
        path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=k_neighbors,
        voxel_size=args.voxel_size,
        use_features=tuple(args.use_features)
    )