import os
import shutil
from sklearn.model_selection import train_test_split
from config import DATASETS_DIR, SEED, BASE_DIR_MINE
# =======================================
# 設定
# =======================================
TRAIN_DIR = BASE_DIR_MINE / "train"
VAL_DIR   = BASE_DIR_MINE / "val"
TEST_DIR  = BASE_DIR_MINE / "test"

FOLDERS = ["plant", "inside", "outside"]  # フォルダごとに分けるシーン
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED = 42  # 再現性のため

# =======================================
# フォルダ作成
# =======================================
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# =======================================
# 分割・コピー関数
# =======================================
def split_and_copy(src_dir, dst_train, dst_val, dst_test):
    # .npy ファイルを取得
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(".npy")]
    
    # train / temp に分割
    X_train, X_temp = train_test_split(files, test_size=VAL_RATIO + TEST_RATIO, random_state=SEED, shuffle=True)
    
    # temp → val / test に分割
    X_val, X_test = train_test_split(X_temp, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=SEED, shuffle=True)
    
    # コピー
    for f in X_train:
        shutil.copy(f, os.path.join(dst_train, os.path.basename(f)))
    for f in X_val:
        shutil.copy(f, os.path.join(dst_val, os.path.basename(f)))
    for f in X_test:
        shutil.copy(f, os.path.join(dst_test, os.path.basename(f)))
    
    # 分割数を返す（確認用）
    return len(X_train), len(X_val), len(X_test)

# =======================================
# フォルダごとに分割・コピー
# =======================================
total_counts = {}
for folder in FOLDERS:
    src = os.path.join(DATASETS_DIR, folder)
    counts = split_and_copy(src, TRAIN_DIR, VAL_DIR, TEST_DIR)
    total_counts[folder] = counts

# =======================================
# 結果表示
# =======================================
for folder, counts in total_counts.items():
    print(f"{folder} → Train: {counts[0]}, Val: {counts[1]}, Test: {counts[2]}")

# 合計
train_total = sum(counts[0] for counts in total_counts.values())
val_total   = sum(counts[1] for counts in total_counts.values())
test_total  = sum(counts[2] for counts in total_counts.values())
print(f"合計 → Train: {train_total}, Val: {val_total}, Test: {test_total}")
