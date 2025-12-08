import os
import shutil
import json
from sklearn.model_selection import train_test_split, KFold
from config import DATASETS_DIR, BASE_DIR_MINE, SEED

# =======================================
# 設定
# =======================================
TRAIN_DIR = BASE_DIR_MINE / "train"
TEST_DIR  = BASE_DIR_MINE / "test"
FOLDERS = ["plant", "inside", "outside"]
TRAIN_RATIO = 0.9
TEST_RATIO  = 0.1
N_FOLDS = 5  # k-fold

# =======================================
# フォルダ作成
# =======================================
for d in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# =======================================
# 分割・コピー関数
# =======================================
def split_and_copy(src_dir, dst_train, dst_test, test_ratio=TEST_RATIO):
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(".npy")]
    X_train, X_test = train_test_split(files, test_size=test_ratio, random_state=SEED, shuffle=True)
    
    for f in X_train:
        shutil.copy(f, os.path.join(dst_train, os.path.basename(f)))
    for f in X_test:
        shutil.copy(f, os.path.join(dst_test, os.path.basename(f)))
    
    return X_train, X_test

# =======================================
# フォルダごとに分割・コピー
# =======================================
train_files_total = []
test_files_total  = []

for folder in FOLDERS:
    src = os.path.join(DATASETS_DIR, folder)
    X_train, X_test = split_and_copy(src, TRAIN_DIR, TEST_DIR)
    train_files_total.extend(X_train)
    test_files_total.extend(X_test)
    print(f"{folder} → Train: {len(X_train)}, Test: {len(X_test)}")

print(f"合計 → Train: {len(train_files_total)}, Test: {len(test_files_total)}")

# =======================================
# k-fold 用の train/val 分割
# =======================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# X_train を k-fold 分割して train/val を作成
folds = []
# 1. k-fold 分割
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    train_files = [X_train[i] for i in train_idx]
    val_files   = [X_train[i] for i in val_idx]
    folds.append((train_files, val_files))
    print(f"Fold {fold_idx+1}: Train {len(train_files)}, Val {len(val_files)}")

# 2. 辞書形式に変換して保存
folds_dict = {}
for i, (train_files, val_files) in enumerate(folds):
    folds_dict[f"fold_{i+1}"] = {
        "train": train_files,
        "val": val_files
    }

folds_save_path = BASE_DIR_MINE / "folds.json"
with open(folds_save_path, "w") as f:
    json.dump(folds_dict, f, indent=2)

print(f"Saved all folds to {folds_save_path}")
