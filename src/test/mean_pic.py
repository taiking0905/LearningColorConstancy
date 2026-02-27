import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================
# 設定
# =============================
# img_dir = r"D:\ColorConstancy\masking\iphone"
img_dir = r"E:\ColorConstancy\masking\Gehler'sRawDataset"

COLORS = {
    "8D": "red",
    "IMG_0": "blue",
    "IMG_4": "green",
}

# =============================
# ファイル取得
# =============================
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]

# =============================
# 結果保存
# =============================
results = defaultdict(
    lambda: {
        "mean_rgb": [],
        "var_rgb": [],
        "mean_rg": [],
        "var_rg": [],
        "mean_gb": [],
        "var_gb": [],
    }
)

# =============================
# 集計処理
# =============================
for file in img_files:

    # ---- デバイス判定 ----
    if file.startswith("8D"):
        dev = "8D"
    elif file.startswith("IMG_0"):
        dev = "IMG_0"
    elif file.startswith("IMG_4"):
        dev = "IMG_4"
    else:
        continue

    path = os.path.join(img_dir, file)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)

    img_flat = img.reshape(-1, 3)

    # ---- ピクセル正規化 ----
    pixel_sum = img_flat.sum(axis=1, keepdims=True)
    pixel_sum[pixel_sum == 0] = 1
    img_norm = img_flat / pixel_sum

    # ---- RGB統計 ----
    mean_rgb = img_norm.mean(axis=0)
    var_rgb = img_norm.var(axis=0)

    # ---- RG ----
    rg = img_norm[:, [0, 1]]
    mean_rg = rg.mean(axis=0)
    var_rg = rg.var(axis=0)

    # ---- GB ----
    gb = img_norm[:, [1, 2]]
    mean_gb = gb.mean(axis=0)
    var_gb = gb.var(axis=0)

    # ---- 保存 ----
    results[dev]["mean_rgb"].append(mean_rgb)
    results[dev]["var_rgb"].append(var_rgb)
    results[dev]["mean_rg"].append(mean_rg)
    results[dev]["var_rg"].append(var_rg)
    results[dev]["mean_gb"].append(mean_gb)
    results[dev]["var_gb"].append(var_gb)

# =============================
# 最終統計
# =============================
final_stats = {}

for dev, data in results.items():
    final_stats[dev] = {}

    for key in ["rgb", "rg", "gb"]:
        mean_all = np.mean(data[f"mean_{key}"], axis=0)
        var_all = np.mean(data[f"var_{key}"], axis=0)
        cv_all = np.sqrt(var_all) / mean_all

        final_stats[dev][key] = {
            "mean": mean_all,
            "var": var_all,
            "cv": cv_all,
        }

# =============================
# 出力
# =============================
for dev, stats in final_stats.items():
    print(f"\n=== Device: {dev} ===")
    for key in ["rgb", "rg", "gb"]:
        print(f"{key.upper()} mean: {stats[key]['mean']}")
        print(f"{key.upper()} variance: {stats[key]['var']}")
        print(f"{key.upper()} CV: {stats[key]['cv']}")

# =============================
# 3D RGBプロット
# =============================
def plot_rgb_3d(results):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for dev, data in results.items():
        rgb_means = np.array(data["mean_rgb"])

        ax.scatter(
            rgb_means[:, 0],
            rgb_means[:, 1],
            rgb_means[:, 2],
            label=dev,
            alpha=0.6,
            s=10,
            color=COLORS.get(dev, "black"),
        )

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title("RGB Mean Distribution")
    ax.legend()
    plt.show()


# =============================
# RG 2Dプロット
# =============================
def plot_rg(results):
    plt.figure(figsize=(7, 7))

    for dev, data in results.items():
        rg_means = np.array(data["mean_rg"])

        plt.scatter(
            rg_means[:, 0],
            rg_means[:, 1],
            label=dev,
            alpha=0.6,
            s=10,
            color=COLORS.get(dev, "black"),
        )

    plt.xlabel("R")
    plt.ylabel("G")
    plt.title("RG Mean Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================
# GB 2Dプロット
# =============================
def plot_gb(results):
    plt.figure(figsize=(7, 7))

    for dev, data in results.items():
        gb_means = np.array(data["mean_gb"])

        plt.scatter(
            gb_means[:, 0],
            gb_means[:, 1],
            label=dev,
            alpha=0.6,
            s=10,
            color=COLORS.get(dev, "black"),
        )

    plt.xlabel("G")
    plt.ylabel("B")
    plt.title("GB Mean Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================
# プロット実行
# =============================
plot_rgb_3d(results)
plot_rg(results)
plot_gb(results)
