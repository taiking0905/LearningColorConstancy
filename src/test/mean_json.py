import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# =============================
# 設定
# =============================
JSON_PATH = "E:/ColorConstancy/real_rgb.json"

DEVICE_PREFIXES = {
    "8D": "8D",
    "IMG_0": "IMG_0",
    "IMG_4": "IMG_4",
}

COLORS = {
    "8D": "red",
    "IMG_0": "blue",
    "IMG_4": "green",
}


# =============================
# データ読み込み
# =============================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =============================
# デバイス判定
# =============================
def detect_device(filename):
    for prefix, dev in DEVICE_PREFIXES.items():
        if filename.startswith(prefix):
            return dev
    return None


# =============================
# RGB集計
# =============================
def collect_rgb(data):
    devices = defaultdict(lambda: {"R": [], "G": [], "B": []})

    for entry in data:
        dev = detect_device(entry["filename"])
        if dev is None:
            continue

        rgb = entry["real_rgb"]
        total = sum(rgb)
        if total == 0:
            continue

        r, g, b = [c / total for c in rgb]

        devices[dev]["R"].append(r)
        devices[dev]["G"].append(g)
        devices[dev]["B"].append(b)

    return devices


# =============================
# 3Dプロット
# =============================
def plot_3d(devices):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for dev, vals in devices.items():
        ax.scatter(
            vals["R"],
            vals["G"],
            vals["B"],
            label=dev,
            alpha=0.6,
            s=10,
            color=COLORS.get(dev, "black"),
        )

    ax.set_xlabel("R ratio")
    ax.set_ylabel("G ratio")
    ax.set_zlabel("B ratio")
    ax.legend()
    ax.set_title("Device RGB Ratio Distribution")
    plt.show()


# =============================
# G基準プロット
# =============================
def plot_g_normalized(devices):
    plt.figure(figsize=(7, 7))
    eps = 1e-8

    for dev, vals in devices.items():
        R = np.array(vals["R"])
        G = np.array(vals["G"])
        B = np.array(vals["B"])

        Rg = R / (G + eps)
        Bg = B / (G + eps)

        plt.scatter(
            Rg,
            Bg,
            label=dev,
            alpha=0.6,
            s=10,
            color=COLORS.get(dev, "black"),
        )

    plt.xlabel("R / G")
    plt.ylabel("B / G")
    plt.title("Device Distribution (G normalized)")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================
# main
# =============================
def main():
    data = load_json(JSON_PATH)
    devices = collect_rgb(data)

    plot_3d(devices)
    plot_g_normalized(devices)


if __name__ == "__main__":
    main()
