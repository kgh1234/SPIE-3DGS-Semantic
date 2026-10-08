import torch
import numpy as np

def update_3dgs_attributes(points_3dgs, points_pointcept, colors_pointcept, normals_pointcept,
                           labels_pointcept, labels200_pointcept, instances_pointcept,
                           k_neighbors=10, use_label_consistency=True, ignore_threshold=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 입력 텐서 변환
    p3dgs = torch.from_numpy(points_3dgs).float().to(device)
    ppcpt = torch.from_numpy(points_pointcept).float().to(device)

    dists = torch.cdist(p3dgs, ppcpt)
    knn_idx = torch.topk(dists, k=k_neighbors, largest=False).indices

    # 속성 변환
    colors_src = torch.from_numpy(colors_pointcept).to(device)
    normals_src = torch.from_numpy(normals_pointcept).float().to(device)
    labels_src = torch.from_numpy(labels_pointcept).long().to(device)
    labels200_src = torch.from_numpy(labels200_pointcept).long().to(device)
    inst_src = torch.from_numpy(instances_pointcept).long().to(device)

    neighbor_colors = colors_src[knn_idx]
    neighbor_normals = normals_src[knn_idx]
    neighbor_labels = labels_src[knn_idx]
    neighbor_labels200 = labels200_src[knn_idx]
    neighbor_instances = inst_src[knn_idx]

    # 평균 색상 및 법선
    colors_3dgs = neighbor_colors.float().mean(dim=1).byte().cpu().numpy()
    normals_3dgs = torch.nn.functional.normalize(neighbor_normals.float().mean(dim=1), dim=1).cpu().numpy()

    def vote(neighbors, ignore_val=-1):
        valid_mask = neighbors != ignore_val
        ignore_ratio = 1.0 - valid_mask.float().mean(dim=1)
        vote_base = torch.where(valid_mask, neighbors, torch.zeros_like(neighbors))
        out_labels = []
        for i in range(vote_base.shape[0]):
            counts = torch.bincount(vote_base[i].cpu(), minlength=300)
            out_labels.append(torch.argmax(counts))
        out_labels = torch.stack(out_labels)
        ignore_mask = (ignore_ratio >= ignore_threshold).cpu()
        out_labels[ignore_mask] = ignore_val
        return out_labels.cpu().numpy()

    labels_3dgs = vote(neighbor_labels)
    labels200_3dgs = vote(neighbor_labels200)
    instances_3dgs = vote(neighbor_instances)

    if use_label_consistency:
        mask = []
        for i in range(neighbor_labels.shape[0]):
            row = neighbor_labels[i].cpu()
            valid = row[row != -1]
            if len(valid) == 0:
                mask.append(False)
                continue
            counts = torch.bincount(valid, minlength=300)
            ratio = counts.max().item() / k_neighbors
            mask.append(ratio >= ignore_threshold)
        mask = np.array(mask)
    else:
        mask = np.ones(len(points_3dgs), dtype=bool)

    return colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask
