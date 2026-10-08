import os
import json
import argparse
import numpy as np
from sklearn.neighbors import KDTree
from collections import Counter
from tqdm import tqdm

# ===================== 라벨 보완 함수 =====================
def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir,
                               k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02):
    from gp_utils import load_pointcept_data, load_3dgs_data, voxelize_3dgs
    from fusion_utils import preprocess_3dgs_attributes
    if args.use_gpu:
        from gp_utils_gpu import update_3dgs_attributes
    else:
        from gp_utils import update_3dgs_attributes

    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    colors_pointcept = pointcept_data['color']
    normals_pointcept = pointcept_data['normal']
    labels_pointcept = pointcept_data['segment20']
    labels200_pointcept = pointcept_data['segment200']
    instances_pointcept = pointcept_data['instance']

    points_3dgs, normals_3dgs, raw_features_3dgs = load_3dgs_data(path_3dgs)
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)

    if voxel_size > 0:
        points_3dgs, _ = voxelize_3dgs(points_3dgs, features_3dgs, voxel_size, k_neighbors)

    colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask = update_3dgs_attributes(
        points_3dgs, points_pointcept, colors_pointcept, normals_pointcept,
        labels_pointcept, labels200_pointcept, instances_pointcept,
        k_neighbors=k_neighbors, use_label_consistency=True,
        ignore_threshold=ignore_threshold
    )

    uncertainty = np.where(mask, 0.0, 1.0).astype(np.float32)

    # 보완
    labels_3dgs, uncertainty = refine_uncertain_labels_with_entropy(
        points_3dgs, labels_3dgs, uncertainty,
        K=6, entropy_threshold=0.7
    )

    valid_mask = (labels_3dgs != -1) & (uncertainty < 0.5)
    points_3dgs = points_3dgs[valid_mask]
    colors_3dgs = colors_3dgs[valid_mask]
    normals_3dgs = normals_3dgs[valid_mask]
    labels_3dgs = labels_3dgs[valid_mask]
    labels200_3dgs = labels200_3dgs[valid_mask]
    instances_3dgs = instances_3dgs[valid_mask]
    uncertainty = uncertainty[valid_mask]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'coord.npy'), points_3dgs.astype(np.float32))
    np.save(os.path.join(output_dir, 'color.npy'), colors_3dgs.astype(np.uint8))
    np.save(os.path.join(output_dir, 'normal.npy'), normals_3dgs.astype(np.float32))
    np.save(os.path.join(output_dir, 'segment20.npy'), labels_3dgs.astype(np.int64))
    np.save(os.path.join(output_dir, 'segment200.npy'), labels200_3dgs.astype(np.int64))
    np.save(os.path.join(output_dir, 'instance.npy'), instances_3dgs.astype(np.int64))
    np.save(os.path.join(output_dir, 'uncertainty.npy'), uncertainty.reshape(-1, 1).astype(np.float32))

def process_single_scene(scene, split, input_root, output_root, path_3dgs_root,
                         k_neighbors=5, voxel_size=0.02):
    try:
        pointcept_dir = os.path.join(input_root, split, scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        output_dir = os.path.join(output_root, split, scene)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"[SKIP] Pointcept data missing: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"[SKIP] 3DGS PLY missing: {path_3dgs}")
            return

        merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir,
                                  k_neighbors=k_neighbors, ignore_threshold=0.6, voxel_size=voxel_size)
    except Exception as e:
        print(f"[ERROR] scene {scene}: {e}")

def process_scenes(input_root, output_root, split, scene_list, path_3dgs_root,
                   k_neighbors=5, num_workers=1, voxel_size=0.02):
    print(f"[{split.upper()}] 총 {len(scene_list)}개 scene 처리 중...")
    for scene in tqdm(scene_list, desc=f"Processing {split}", unit="scene"):
        process_single_scene(scene, split, input_root, output_root,
                             path_3dgs_root, k_neighbors, voxel_size=voxel_size)

def compute_entropy(label_counts):
    total = sum(label_counts.values())
    probs = np.array([count / total for count in label_counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-8))

def refine_uncertain_labels_with_entropy(
    points_3dgs, labels_3dgs, uncertainty,
    K=6, entropy_threshold=0.7
):
    certified_mask = uncertainty <= 0.6
    certified_points = points_3dgs[certified_mask]
    certified_labels = labels_3dgs[certified_mask]
    tree = KDTree(certified_points)

    uncertain_mask = (labels_3dgs == -1) & (uncertainty > 0.5)
    candidate_indices = np.where(uncertain_mask)[0]
    num_fixed = 0

    for idx in tqdm(candidate_indices, desc="Refining uncertain labels"):
        query_point = points_3dgs[idx].reshape(1, -1)
        dists, knn_idx = tree.query(query_point, k=K)

        neighbor_labels = certified_labels[knn_idx[0]]
        label_counts = Counter(neighbor_labels)
        entropy = compute_entropy(label_counts)

        if entropy < entropy_threshold:
            labels_3dgs[idx] = label_counts.most_common(1)[0][0]
            uncertainty[idx] = 0.5
            num_fixed += 1

    print(f"[Refinement] 총 {num_fixed}개의 라벨을 보완함 (K={K}, entropy<thresh={entropy_threshold})")
    return labels_3dgs, uncertainty

# ===================== argparse 먼저 선언 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--output_root", default="...", help="Output directory")
parser.add_argument("--data_type", default="samples100", choices=["full", "samples100"])
parser.add_argument("--num_workers", default=1, type=int)
parser.add_argument("--voxel_size", default=0.04, type=float)
parser.add_argument("--use_gpu", action="store_true")
args = parser.parse_args()

# ===================== 실행 진입점 =====================
if __name__ == "__main__":
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_root = config['input_root']
    output_root = args.output_root
    path_3dgs_root = config['path_3dgs_root']
    meta_root = config['meta_root']
    k_neighbors = 5

    if args.data_type == 'full':
        train_meta_file = "scannetv2_train.txt"
        val_meta_file = "scannetv2_val.txt"
    else:
        train_meta_file = "train100_samples.txt"
        val_meta_file = "valid20_samples.txt"

    with open(os.path.join(meta_root, train_meta_file)) as f:
        train_scenes = f.read().splitlines()
    process_scenes(input_root, args.output_root, 'train', train_scenes,
                   path_3dgs_root, k_neighbors, args.num_workers, args.voxel_size)

    with open(os.path.join(meta_root, val_meta_file)) as f:
        val_scenes = f.read().splitlines()
    process_scenes(input_root, args.output_root, 'val', val_scenes,
                   path_3dgs_root, k_neighbors, args.num_workers, args.voxel_size)
