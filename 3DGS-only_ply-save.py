import os
import argparse
import numpy as np
from tqdm import tqdm
import json

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data, voxelize_3dgs, update_3dgs_attributes, save_ply
from fusion_utils import preprocess_3dgs_attributes

def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, exp, k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02):
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

    # 4. 3DGS 데이터 Voxelization (옵션)
    voxelize = voxel_size != 0  # voxel_size가 0이 아니면 voxelize 활성화
    if voxelize:
        print("Applying Voxelization to 3DGS points...")
        points_3dgs, features_3dgs = voxelize_3dgs(
            points_3dgs, features_3dgs, voxel_size=voxel_size, k_neighbors=5)
    
    # 5. Point cloud color, normals, labels transfer 
    print("Updating 3DGS attributes from Pointcept...")
    colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask = update_3dgs_attributes(
        points_3dgs, points_pointcept, colors_pointcept, normals_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept,
        k_neighbors=k_neighbors, use_label_consistency=True, ignore_threshold=ignore_threshold
    )

    # 6. 라벨 일관성 필터링 적용
    points_3dgs = points_3dgs[mask]
    colors_3dgs = colors_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    labels_3dgs = labels_3dgs[mask]
    labels200_3dgs = labels200_3dgs[mask]
    instances_3dgs = instances_3dgs[mask]
    features_3dgs = features_3dgs[mask]  # 3DGS 속성도 필터링하여 동기화

    # 7. -1 라벨 3DGS 점 비율 출력
    ignore_count = np.sum(labels_3dgs == -1)
    total_count = len(labels_3dgs)
    ignore_ratio = ignore_count / total_count if total_count > 0 else 0
    print(f"3DGS points with label -1: {ignore_count}/{total_count} (Ratio: {ignore_ratio:.4f})")

    # 8. PLY 파일로 저장
    print(f"Saving 3DGS points to PLY...")
    output_dir_path = os.path.join(output_dir, exp)
    os.makedirs(output_dir_path, exist_ok=True)
    output_ply_path = os.path.join(output_dir_path, f"{exp}.ply")
    save_ply(
        points_3dgs, colors_3dgs, labels_3dgs, output_ply_path,
        save_separate_labels=True,  # 라벨 색상 기반 별도 파일 생성
        points_pointcept=points_pointcept,
        colors_pointcept=colors_pointcept,
    )
    #print(f"Saved 3DGS PLY to {output_ply_path}")


def process_single_scene(scene, input_root, output_root, exp, path_3dgs_root, k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02):
    """
    단일 scene을 처리하여 PLY 파일로 저장.
    
    Args:
        scene (str): 처리할 scene 이름.
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리.
        output_root (str): 출력 디렉토리.
        exp (str): 실험 이름.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        k_neighbors (int): 라벨 복사 및 속성 전달에 사용할 이웃 점 개수.
        ignore_threshold (float): 라벨 일관성 필터링 임계값.
        voxel_size (float): Voxel 크기 (기본값: 0.02m).
    """
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

        merge_pointcept_with_3dgs(
            pointcept_dir, path_3dgs, output_dir, exp,
            k_neighbors=k_neighbors, ignore_threshold=ignore_threshold, voxel_size=voxel_size
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single ScanNet scene with 3DGS and save as PLY for visualization.")
    parser.add_argument(
        "--scene",
        default="scene0011_00",
        help="Scene to process (e.g., scene0011_00)",
    )
    parser.add_argument(
        "--output_root",
        default="./outputs/test_samples10",
        help="Output path for processed data",
    )
    parser.add_argument(
        "--exp",
        default="vox002_3dgs_with_labels",
        help="Experiment name",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.02,
        type=float,
        help="Voxel size for 3DGS voxelization (default: 0.02m).",
    )
    parser.add_argument(
        "--ignore_threshold",
        default=0.6,
        type=float,
        help="Threshold for label consistency filtering.",
    )

    args = parser.parse_args()

    # Config 파일 로드
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_root = config['input_root']
    path_3dgs_root = config['path_3dgs_root']
    k_neighbors = 5

    # 단일 Scene 처리
    process_single_scene(
        args.scene,
        input_root,
        args.output_root,
        args.exp,
        path_3dgs_root,
        k_neighbors=k_neighbors,
        ignore_threshold=args.ignore_threshold,
        voxel_size=args.voxel_size
    )