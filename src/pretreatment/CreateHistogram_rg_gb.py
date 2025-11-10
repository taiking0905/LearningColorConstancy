import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import numpy as np
import os
import pandas as pd

from config import move_figure, to_8bit_gamma
from utils.load_and_normalize_image import load_and_normalize_image

# =======================================
# 定数設定
# =======================================
# rg平面のビン幅（R/(R+G+B), G/(R+G+B)を区切る幅）
bin_width = 0.02
# 0〜1をbin_widthで分割したときのビン数
num_bins = int(1.0 / bin_width)
# CNNに入れるためのサイズresize
size =224

def compute_2d_histograms(rgb_normalized, valid_mask):
    r, g, b = rgb_normalized[..., 0], rgb_normalized[..., 1], rgb_normalized[..., 2]
    num_bins = int(1.0 / bin_width)

    r_bins = np.clip((r[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    g_bins = np.clip((g[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    b_bins = np.clip((b[valid_mask] / bin_width).astype(int), 0, num_bins - 1)

    hist_rg = np.zeros((num_bins, num_bins), dtype=np.uint32)
    hist_gb = np.zeros((num_bins, num_bins), dtype=np.uint32)

    for r_bin, g_bin in zip(r_bins, g_bins):
        hist_rg[g_bin, r_bin] += 1
    for g_bin, b_bin in zip(g_bins, b_bins):
        hist_gb[b_bin, g_bin] += 1

    return hist_rg, hist_gb

def normalize_histogram(hist):
    if hist.max() > 0:
        return (hist / hist.max()).astype(np.float32)
    else:
        return hist.astype(np.float32)

def combine_rg_gb_histograms(presence_rg, presence_gb):
    UINT8_MAX = 255  # 8bit画像の最大値
    # INTER_LINEARよりもINTER_NEARESTのほうがいいかも
    rg_upsampled = cv2.resize(presence_rg, (size, size), interpolation=cv2.INTER_LINEAR)
    gb_upsampled = cv2.resize(presence_gb, (size, size), interpolation=cv2.INTER_LINEAR)

    rg_8bit = (rg_upsampled * UINT8_MAX).astype(np.uint8)
    gb_rotated = np.rot90((gb_upsampled * UINT8_MAX).astype(np.uint8), 2)

    combined = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if x + y <= size:
                combined[y, x] = rg_8bit[y, x]
            else:
                combined[y, x] = gb_rotated[y, x]
    return combined, np.stack([combined], axis=0)

def plot_2d_histogram(hist, title, xlabel, ylabel, filename, cmap='viridis', logscale=True):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    norm = LogNorm(vmin=1, vmax=255) if logscale else None
    im = ax.imshow(hist.T, origin='lower', cmap=cmap,
                   extent=[0,1,0,1], aspect='auto', norm=norm)
    ax.set_title(f"{title}: {filename}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.colorbar(im, ax=ax, label='Pixel Count' if logscale else 'Value')
    return fig

def CreateHistogram_rg_gb(image_path, output_path):
    filename, _, _, rgb_normalized , valid_mask=load_and_normalize_image(image_path)

    # ==============================
    # 2D ヒストグラム（rg空間 &gb空間）
    # ==============================
    
    hist_rg_2d, hist_gb_2d = compute_2d_histograms(rgb_normalized, valid_mask)

    presence_rg = normalize_histogram(hist_gb_2d)
    presence_gb = normalize_histogram(hist_gb_2d)

    combined, stacked = combine_rg_gb_histograms(presence_rg, presence_gb)

    np.save(os.path.join(output_path, f"{filename}.npy"), stacked)
    print(f"Saved upsampled 224x224x2 histogram to: {filename}.npy")

    fig1 = plot_2d_histogram(hist_rg_2d, "2D Hist (rg count)", 'r = R/(R+G+B)', 'g = G/(R+G+B)', filename)
    fig2 = plot_2d_histogram(hist_gb_2d, "2D Hist (gb count)", 'g = G/(R+G+B)', 'b = B/(R+G+B)', filename)
    fig3 = plot_2d_histogram(combined, "RG & GB Combined Histogram", '', '', filename, logscale=True)

    # ======================
    # 共通イベントハンドラで同時に閉じる
    # ======================

    # 左端・右端に配置（モニタサイズに応じて調整）
    move_figure(fig1, 0, 20)        # 左端（x=0, y=100）
    move_figure(fig2, 30, 20)     # 右寄り（x=1200, y=100）※必要に応じて調整
    move_figure(fig3, 60, 20)

    # ========= Enterで同時に閉じる =========
    def on_key(event):
        if event.key == ' ':
            for f in [fig1, fig2, fig3]:
                plt.close(f)

    for f in [fig1, fig2, fig3]:
        f.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=True)

