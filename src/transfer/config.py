# config.py
from pathlib import Path
import torch
import numpy as np
import random
import psutil
import os

# -------------------------------
# パス設定
# -------------------------------
_base_dir = None

def find_drive_with_folder(folder_name="ColorConstancy"):
    for part in psutil.disk_partitions():
        if 'removable' in part.opts.lower():
            drive = part.mountpoint
            if os.path.exists(os.path.join(drive, folder_name)):
                return drive
    return None

def get_base_dir():
    global _base_dir
    if _base_dir is None:
        drive = find_drive_with_folder("ColorConstancy")
        if not drive:
            raise RuntimeError("ドライブが見つかりません")
        _base_dir = Path(drive) / "ColorConstancy/TransferDataset"
    return _base_dir

BASE_DIR = get_base_dir()
LCC_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
REAL_RGB_JSON_PATH = (BASE_DIR / "..") / "real_rgb.json"
TEST_DIR = BASE_DIR / "test"
PRE_TEST_DIR = BASE_DIR / "pretest"
OUTPUT_DIR = LCC_DIR / "outputs"
# -------------------------------
# ハイパーパラメータ
# -------------------------------
EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE =  1e-4
WEIGHT = 5e-5
DROPOUT = 0.5

# -------------------------------
# 設定パラメータ
# -------------------------------
SEED = 42
OUTPUT_DIM = 3
ERASE_PROB =0.5
ERASE_SIZE = 60


FREEZE_MODE = "fc_only"  
# -------------------------------
# none 全層学習
# fc_only FCだけ学習
# layer4_fc 4層以降を学習
# deep_blocks 3層以降を学習
# freeze_only_layer conv1とFC層を学習
# -------------------------------

# -------------------------------
# デバイス設定
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# ランダムシード固定
# -------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 追加: 完全な再現性のための設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


