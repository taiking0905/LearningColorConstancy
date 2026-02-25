from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import psutil
import os
import tkinter as tk
import numpy as np

# ======== Matplotlib設定 ========
matplotlib.use('TkAgg')  # TkAggバックエンドを使用（ウィンドウ移動対応）

# ======== グローバル定数 ========

def find_drive_with_folder(folder_name="target_folder"):
    for part in psutil.disk_partitions():
        if 'removable' in part.opts.lower():  # USBなどに限定したいとき
            drive = part.mountpoint
            if os.path.exists(os.path.join(drive, folder_name)):
                return drive
    return None

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
        _base_dir = Path(drive) / "ColorConstancy"
    return _base_dir

# データパスのベース
BASE_PATH = get_base_dir()

# カメラ設定（black/whiteレベル）
BLACK_LEVEL = 0
WHITE_LEVEL = 4095

# ======== ディレクトリ構成 ========

def setup_directories():
    """
    必要なディレクトリを作成し、パス辞書を返す。
    real_rgb.json の存在チェックも行う。
    """
    dirs = {
        "INPUT": BASE_PATH / "datasets",
        "MASK": BASE_PATH / "masking",
        "COLORCHECKER": BASE_PATH/ "colorchecker",
        "HIST": BASE_PATH / "histogram",
        "HIST_RG_GB": BASE_PATH / "histogram_rg_gb",
        "TEMP": BASE_PATH / "temp",
        # "TEACHER": BASE_PATH / "teacher",
        # "TEACHER_HIST": BASE_PATH / "teacherhist",
        "REAL_RGB_JSON": BASE_PATH / "real_rgb.json",
        "END": BASE_PATH / "enddatasets",
        "TEST": BASE_PATH / "test"
    }

    # ディレクトリ作成（JSONファイルは除外）
    for key, path in dirs.items():
        if key != "REAL_RGB_JSON":
            path.mkdir(parents=True, exist_ok=True)

    # JSONファイルの存在確認
    if not dirs["REAL_RGB_JSON"].exists():
        print("⚠ real_rgb.json が存在しません。先に作成してください。")

    return dirs

# ======== ウィンドウ配置関数 ========

def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # ウィンドウを表示しない
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


def move_figure(fig, x: int, y: int):
    """
    Matplotlibのウィンドウ位置を画面上の指定座標(x, y)に移動する。
    """
    screen_w, screen_h = get_screen_size()
    x = int(screen_w * x / 100)
    y = int(screen_h * y / 100)
    backend = plt.get_backend()

    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))
    elif backend in ['Qt4Agg', 'Qt5Agg', 'QtAgg']:
        fig.canvas.manager.window.move(x, y)