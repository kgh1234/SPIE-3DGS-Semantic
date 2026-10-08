import os

def count_subfolders(directory):
    if not os.path.exists(directory):
        return -1  # 경로 없을 경우
    return sum(os.path.isdir(os.path.join(directory, name)) for name in os.listdir(directory))

# 고정된 base 디렉토리
base_dir = "./data"

# 실험용 데이터셋 리스트
data_list = [
    "fusion_pd00005_scale02_opa07_all",
]

# 순회하면서 train/val 폴더 내 scene 개수 출력
for dataset_name in data_list:
    dataset_path = os.path.join(base_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue  # 폴더 아님

    print(f"\n📁 [데이터셋] {dataset_name}")
    for split in ["train", "val"]:
        split_path = os.path.join(dataset_path, split)
        count = count_subfolders(split_path)
        if count == -1:
            print(f"    └─ ⚠️  '{split}' 폴더 없음: {split_path}")
        else:
            print(f"    └─ ✅ '{split}' 폴더 내 scene 개수: {count}")