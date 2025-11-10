import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd

from config import move_figure
from pretreatment.utils.image_util import load_image, normalize_image, to_8bit_gamma

# =======================================
# 定数設定
# =======================================
# rg平面のビン幅（R/(R+G+B), G/(R+G+B)を区切る幅）
bin_width = 0.02
# 0〜1をbin_widthで分割したときのビン数
num_bins = int(1.0 / bin_width)

# =======================================
# RGBヒストグラムを作成
# =======================================
def create_rgb_histogram(norm_display, mask, filename):
    """
    各チャンネル（R, G, B）のヒストグラムを計算・描画する。
    """
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    # 各色チャンネルごとにヒストグラムを計算
    for i, color in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([norm_display], [i], mask, [256], [0, 1])
        ax.plot(hist, color=color, label=f"{color.upper()} channel")

    ax.set_title(f"RGB Histogram: {filename}")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel Count")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# =======================================
# rg空間上の2Dヒストグラムを作成
# =======================================
def create_2d_histogram(rgb_ratio, valid_mask, filename):
    """
    rg空間（r=R/(R+G+B), g=G/(R+G+B)）上でピクセルの存在を2Dヒストグラムとして可視化。
    presence_maskはピクセルが存在する位置をTrueで表す。
    """
    # 各チャンネルの比率を取得
    r_ratio = rgb_ratio[:, :, 0]
    g_ratio = rgb_ratio[:, :, 1]

    # rg値をbin幅ごとに区切って離散化（ビン番号化）
    r_bins = np.clip((r_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    g_bins = np.clip((g_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)

    # 空の2D配列（num_bins×num_bins）を作成し、存在するピクセルを1でマーク
    binary_hist_2d = np.zeros((num_bins, num_bins), dtype=np.uint8)
    for r_bin, g_bin in zip(r_bins, g_bins):
        binary_hist_2d[g_bin, r_bin] = 1

    # ピクセルが存在する領域をTrueにしたmaskを作成
    presence_mask = binary_hist_2d > 0

    # ======================
    # 2Dヒストグラムの描画
    # ======================
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        presence_mask.T,
        origin='lower',
        cmap='gray',
        extent=[0, 1, 0, 1],
        aspect='auto'
    )

    ax.set_title(f"2D Hist (rg mask): {filename}")
    ax.set_xlabel('r = R / (R+G+B)')
    ax.set_ylabel('g = G / (R+G+B)')
    ax.grid(True, alpha=0.3)
    fig.colorbar(im, ax=ax, label='Pixel Exists (True/False)')
    return fig, presence_mask

# =======================================
# 2Dヒストグラム（presence mask）をフラット化してCSVに保存
# =======================================
def save_flat_mask(presence_mask, filename, output_path):
    """
    2D presence mask（rg平面上のピクセル分布）を
    1次元ベクトルに変換し、CSVとして保存する。
    """
    flat_mask = []
    for g in range(num_bins):
        for r in range(num_bins):
            # r+g <= 1 の範囲（有効領域）のみを対象とする
            if (r * bin_width + g * bin_width) <= 1.0:
                flat_mask.append(presence_mask[g, r])

    flat_mask = np.array(flat_mask).astype(int)

    # 1行に展開してCSV出力（header/indexなし）
    pd.DataFrame([flat_mask]).to_csv(
        os.path.join(output_path, f"{filename}.csv"), index=False, header=False
    )

# =======================================
# メイン関数：画像からヒストグラムを生成・表示・保存
# =======================================
def CreateHistogram(image_path, output_path):
    """
    画像を読み込み、RGBヒストグラム・rg空間ヒストグラムを作成・表示し、
    rgマスクをCSVに保存する。
    """
    # ======================
    # 画像の読み込み・正規化
    # ======================

    # 画像読み込み
    filename, img = load_image(image_path)

    # 正規化
    rgb_ratio, _, valid_mask = normalize_image(img)

    # 表示用画像
    display_img = to_8bit_gamma(img)


    # 全ピクセルが有効である範囲のマスクを作成（単純に非黒領域）
    mask = cv2.inRange(display_img, (1, 1, 1), (255, 255, 255))

    # OpenCVヒストグラム計算用に正規化
    norm_display = display_img.astype('float32') / 255.0

    # ======================
    # ヒストグラムの作成
    # ======================
    fig1 = create_rgb_histogram(norm_display, mask, filename)
    fig2, presence_mask = create_2d_histogram(rgb_ratio, valid_mask, filename)

    # ======================
    # 図の配置と操作設定
    # ======================
    # 位置を調整（マルチモニタ対応なら調整必要）
    move_figure(fig1, 0, 20)       # 左端
    move_figure(fig2, 60, 20)      # 右寄り

    # スペースキーを押すと両方閉じる
    def on_key(event):
        if event.key == ' ':
            plt.close(fig1)
            plt.close(fig2)

    # どちらのウィンドウでも同じイベントで閉じる
    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)

    # ======================
    # 表示ブロック（終了待ち）
    # ======================
    plt.show(block=True)

    # ======================
    # CSV保存（1250次元 ≒ 有効領域）
    # ======================
    save_flat_mask(presence_mask, filename, output_path)
    print(f"Saved {presence_mask.size}-dim histogram for: {filename}")
