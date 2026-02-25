import os
import cv2
import numpy as np
import json
from config import setup_directories

dirs = setup_directories()
test_dir = dirs["TEST"]

# パラメータ
threshold = 3300     # 白飛び判定境界
max_ratio = 0.01     # 画像比率（1%以下）
real_rgb_limit = 3300  # JSON real_rgb の上限

# === real_rgb JSON 読み込み ===
with open(dirs["REAL_RGB_JSON"], "r") as f:
    real_rgb_data = json.load(f)

# filename → real_rgb の辞書（拡張子なし）
real_rgb_dict = {d["filename"]: d["real_rgb"] for d in real_rgb_data}

# === 4分類 ===
adopted_both = []        # JSON + 画像 OK
adopted_json_only = []   # JSONのみOK
adopted_image_only = []  # 画像のみOK
adopted_none = []        # 両方ダメ

# === TESTフォルダ画像チェック ===
for filename in os.listdir(test_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
        continue

    name_no_ext = os.path.splitext(filename)[0]
    name_base = name_no_ext.replace("_masked", "")

    # ---- JSONのチェック ----
    json_ok = False
    if name_base in real_rgb_dict:
        real_rgb = real_rgb_dict[name_base]
        json_ok = all(v < real_rgb_limit for v in real_rgb)
    else:
        json_ok = False  # JSONにない＝不合格

    # ---- 画像中の白飛び割合チェック ----
    img_path = os.path.join(test_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    exceed_ratio = np.sum(img >= threshold) / img.size
    image_ok = exceed_ratio <= max_ratio

    # ---- 判定 ----
    if json_ok and image_ok:
        adopted_both.append(filename)
    elif json_ok and not image_ok:
        adopted_json_only.append(filename)
    elif image_ok and not json_ok:
        adopted_image_only.append(filename)
    else:
        adopted_none.append(filename)

# === 結果出力 ===
print("\n=== 両方OK ===")
print(adopted_both)
print(f"{len(adopted_both)} 件")

print("\n=== JSONだけOK ===")
print(adopted_json_only)
print(f"{len(adopted_json_only)} 件")

print("\n=== 画像だけOK ===")
print(adopted_image_only)
print(f"{len(adopted_image_only)} 件")

print("\n=== 両方ダメ ===")
print(adopted_none)
print(f"{len(adopted_none)} 件")
