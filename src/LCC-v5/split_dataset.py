import os
import shutil
from sklearn.model_selection import train_test_split
from config import DATASETS_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, SEED

# =======================================
# サイズ設定
# =======================================
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# =======================================
# 画像ファイル取得とラベル分け
# =======================================
all_files = [f for f in os.listdir(DATASETS_DIR) if f.lower().endswith('.npy')]
paths_A = [os.path.join(DATASETS_DIR, f) for f in all_files if f.startswith("8D5U")]
paths_B = [os.path.join(DATASETS_DIR, f) for f in all_files if f.startswith("IMG")]

print("A 枚数:", len(paths_A), "B 枚数:", len(paths_B))

# =======================================
# ドメインごとに train/val/test に分割
# =======================================
def split_domain(paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n_total = len(paths)
    
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val  # 残りを test に
    
    # train / temp
    X_train, X_temp = train_test_split(paths, test_size=n_val + n_test, random_state=SEED)
    # temp → val / test
    X_val, X_test = train_test_split(X_temp, test_size=n_test, random_state=SEED)
    
    return X_train, X_val, X_test


X_train_A, X_val_A, X_test_A = split_domain(paths_A, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
X_train_B, X_val_B, X_test_B = split_domain(paths_B, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

# =======================================
# ドメインを結合
# =======================================
X_train = X_train_A + X_train_B
X_val   = X_val_A   + X_val_B
X_test  = X_test_A  + X_test_B

# =======================================
# フォルダ作成
# =======================================
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# =======================================
# コピー関数
# =======================================
def copy_files(file_list, dest_dir):
    for f in file_list:
        shutil.copy(f, os.path.join(dest_dir, os.path.basename(f)))

copy_files(X_train, TRAIN_DIR)
copy_files(X_val, VAL_DIR)
copy_files(X_test, TEST_DIR)

# =======================================
# 確認
# =======================================
print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))
print("Train A/B:", len(X_train_A), len(X_train_B))
print("Val   A/B:", len(X_val_A), len(X_val_B))
print("Test  A/B:", len(X_test_A), len(X_test_B))
