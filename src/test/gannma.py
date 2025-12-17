import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import cv2
import os

BLACK_LEVEL = 0
WHITE_LEVEL = 4095
LIST = "IMG_0808"

def to_8bit_gamma(img, gamma=2.2):
    img_norm = np.clip((img - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL), 0, 1)
    img_gamma = np.power(img_norm, 1/gamma)
    return (img_gamma * 255).astype(np.uint8)

# 画像読み込み
img_path = r"E:/ColorConstancy/enddatasets/Gehler'sRawDataset"
img_file = os.path.join(img_path, LIST + ".png")
img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
img = np.clip(img - BLACK_LEVEL, 0, None).astype(np.uint16)
img_display = to_8bit_gamma(img)
img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

# 2Dヒストグラム読み込み
npy_path = r"E:\ColorConstancy\histogram_rg_gb\Gehler'sRawDataset"
hist_file = os.path.join(npy_path, LIST + "_masked.npy")
combined = np.load(hist_file)

# 同じウィンドウで左右に並べて表示
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左：画像
axes[0].imshow(img_rgb_display)
axes[0].axis('off')
axes[0].set_title(LIST + " Image")

# 右：2Dヒストグラム
norm = LogNorm(vmin=1, vmax=255)
im = axes[1].imshow(combined.T, origin='lower', cmap='viridis',
                    extent=[0,1,0,1], aspect='auto', norm=norm)
axes[1].set_title(LIST + " RG & GB Histogram")
axes[1].set_xlabel('R-G')
axes[1].set_ylabel('G-B')
axes[1].grid(True, alpha=0.3)
fig.colorbar(im, ax=axes[1], label='Pixel Count')

plt.tight_layout()
plt.show()
