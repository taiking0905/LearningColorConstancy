import os
import shutil
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from config import DATASETS_DIR, BASE_DIR_ORGINAL, SEED

TRAIN_DIR = BASE_DIR_ORGINAL / "train"
TEST_DIR  = BASE_DIR_ORGINAL / "test"
GEHLER_DATASETS_DIR = DATASETS_DIR  / "Gehler'sRawDataset"

TRAIN_RATIO = 0.9
TEST_RATIO  = 0.1
N_FOLDS = 5  # k-fold

# =======================================
# 画像ファイル取得とラベル分け
# =======================================
all_files = [f for f in os.listdir(GEHLER_DATASETS_DIR) if f.lower().endswith('.npy')]
paths_A = [os.path.join(GEHLER_DATASETS_DIR, f) for f in all_files if f.startswith("8D5U")]
paths_B = [os.path.join(GEHLER_DATASETS_DIR, f) for f in all_files if f.startswith("IMG")]

print("A 枚数:", len(paths_A), "B 枚数:", len(paths_B))

# =======================================
# 固定 test と train の分割
# =======================================
def split_fixed(paths, train_ratio):
    n_total = len(paths)
    n_train = int(n_total * train_ratio)
    train_files, test_files = train_test_split(paths, train_size=n_train, random_state=SEED)
    return train_files, test_files

X_train_A, X_test_A = split_fixed(paths_A, TRAIN_RATIO)
X_train_B, X_test_B = split_fixed(paths_B, TRAIN_RATIO)

X_train = X_train_A + X_train_B
X_test  = X_test_A  + X_test_B

# =======================================
# フォルダ作成
# =======================================
for d in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

def copy_files(file_list, dest_dir):
    for f in file_list:
        shutil.copy(f, os.path.join(dest_dir, os.path.basename(f)))

copy_files(X_train, TRAIN_DIR)
copy_files(X_test, TEST_DIR)

print("Train:", len(X_train), "Test:", len(X_test))
print("Train A/B:", len(X_train_A), len(X_train_B))
print("Test  A/B:", len(X_test_A), len(X_test_B))

# =======================================
# k-fold 用の train/val インデックス生成
# =======================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# X_train を k-fold 分割して train/val を作成
folds = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    train_files = [X_train[i] for i in train_idx]
    val_files   = [X_train[i] for i in val_idx]
    folds.append((train_files, val_files))
    print(f"Fold {fold_idx+1}: Train {len(train_files)}, Val {len(val_files)}")
    
folds_save_path = BASE_DIR_ORGINAL / "folds.json"
folds_dict = {f"fold_{i+1}": {"train": train_files, "val": val_files} 
            for i, (train_files, val_files) in enumerate(folds)}

with open(folds_save_path, "w") as f:
    json.dump(folds_dict, f, indent=2)

# folds リストには各 fold の train / val ファイルパスが格納される
# 学習ループ内で fold ごとに使用可能


