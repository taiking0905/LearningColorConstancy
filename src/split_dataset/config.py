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
_base_dir_original = None
_base_dir_mine = None

def find_drive_with_folder(folder_name="ColorConstancy"):
    for part in psutil.disk_partitions():
        if 'removable' in part.opts.lower():
            drive = part.mountpoint
            if os.path.exists(os.path.join(drive, folder_name)):
                return drive
    return None

def get_base_dir():
    global _base_dir_original, _base_dir_mine
    if _base_dir_original is None or _base_dir_mine is None:
        drive = find_drive_with_folder("ColorConstancy")
        if not drive:
            raise RuntimeError("ドライブが見つかりません")
        _base_dir_original = Path(drive) / "ColorConstancy/LCC-v5"
        _base_dir_mine = Path(drive) / "ColorConstancy/TransferDataset"
    return _base_dir_original, _base_dir_mine 

BASE_DIR_ORGINAL, BASE_DIR_MINE = get_base_dir()
LCC_DIR = Path(__file__).resolve().parent
DATASETS_DIR = (BASE_DIR_ORGINAL / "..") / "histogram_rg_gb/"



# -------------------------------
# 設定パラメータ
# -------------------------------
SEED = 42
OUTPUT_DIM = 3
ERASE_PROB =0.6
ERASE_SIZE = 30



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
