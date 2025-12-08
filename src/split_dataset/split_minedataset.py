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
folds = []
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_files_total)):
    train_fold = [train_files_total[i] for i in train_idx]
    val_fold   = [train_files_total[i] for i in val_idx]
    folds.append({
        "train": train_fold,
        "val": val_fold
    })
    print(f"Fold {fold_idx+1}: Train {len(train_fold)}, Val {len(val_fold)}")

# =======================================
# folds.json 保存
# =======================================
folds_save_path = BASE_DIR_MINE / "folds.json"
with open(folds_save_path, "w") as f:
    json.dump(folds, f, indent=2)

