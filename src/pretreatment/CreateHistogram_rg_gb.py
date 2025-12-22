import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
import os

from config import move_figure
from image_util import load_image, normalize_image

# =======================================
# 定数設定
# =======================================
# rg平面のビン幅（R/(R+G+B), G/(R+G+B)を区切る幅）
bin_width = 0.02
# 0〜1をbin_widthで分割したときのビン数
num_bins = int(1.0 / bin_width)
# CNN入力用のリサイズ後のサイズ（224x224）
size = 224


# =======================================
# 2次元ヒストグラムを計算
# =======================================
def compute_2d_histograms(rgb_normalized, valid_mask):
    """
    R,G,B値を正規化した画像から、(r,g) および (g,b) 空間の2次元ヒストグラムを作成する。
    各画素の R/(R+G+B) および G/(R+G+B) などの比率に基づき、binごとの出現回数をカウント。

    Parameters:
        rgb_normalized : np.ndarray
            正規化済みRGB画像 (float, 0〜1)
        valid_mask : np.ndarray
            有効画素(True/False)を示すマスク

    Returns:
        hist_rg, hist_gb : np.ndarray
            rg空間とgb空間のヒストグラム
    """
    # チャンネルごとに分離
    r, g, b = rgb_normalized[..., 0], rgb_normalized[..., 1], rgb_normalized[..., 2]
    num_bins = int(1.0 / bin_width)

    # 各画素をビンインデックスに変換（0〜num_bins-1）
    r_bins = np.clip((r[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    g_bins = np.clip((g[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    b_bins = np.clip((b[valid_mask] / bin_width).astype(int), 0, num_bins - 1)

    # ヒストグラム用の空配列を初期化
    hist_rg = np.zeros((num_bins, num_bins), dtype=np.uint32)
    hist_gb = np.zeros((num_bins, num_bins), dtype=np.uint32)

    # (r,g) 組と (g,b) 組でカウント
    for r_bin, g_bin in zip(r_bins, g_bins):
        hist_rg[g_bin, r_bin] += 1
    for g_bin, b_bin in zip(g_bins, b_bins):
        hist_gb[b_bin, g_bin] += 1

    return hist_rg, hist_gb


# =======================================
# ヒストグラムを0〜1に正規化
# =======================================
def normalize_histogram_for_imagenet(hist):
    """
    ImageNet転移学習用:
    - ヒストグラムを0〜1にスケーリング
    - 微小値カット optional
    """
    hist = hist.astype(np.float32)
    max_val = hist.max()
    if max_val > 0:
        hist /= max_val
    return hist

def combine_rg_gb_histograms(hist_rg, hist_gb, size=224, sigma=1.5):
    """
    rgとgbの2Dヒストグラムを結合してCNN入力用画像と可視化用画像を作成
    ImageNet転移学習対応
    """
    # 1. リサイズ
    rg_resized = cv2.resize(hist_rg.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    gb_resized = cv2.resize(hist_gb.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)

    # 2. Gaussian 平滑化
    rg_smooth = gaussian_filter(rg_resized, sigma=sigma)
    gb_smooth = gaussian_filter(gb_resized, sigma=sigma)

    # 3. 下三角用に180°回転
    gb_rotated = np.rot90(gb_smooth, 2)

    # 4. 上三角: rg / 下三角: gb を結合
    combined_float = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if x + y <= size:
                combined_float[y, x] = rg_smooth[y, x]
            else:
                combined_float[y, x] = gb_rotated[y, x]

    # 5. ノイズ除去（微小値カット）
    low_threshold = combined_float.max() * 1e-5
    combined_float[combined_float < low_threshold] = 0

    # 6. 0〜1スケーリング（max正規化）
    combined_norm = normalize_histogram_for_imagenet(combined_float)

    # 7. CNN入力用: 1チャンネル float32
    combined_for_model = np.expand_dims(combined_norm, axis=0)  # shape: (1, H, W)

    # 8. 可視化用: 0~255スケール
    combined_visual = np.clip(combined_norm * 255.0, 0, 255).astype(np.uint8)

    return combined_for_model, combined_visual


# =======================================
# 2Dヒストグラムを可視化・保存
# =======================================
def plot_2d_histogram(hist, title, xlabel, ylabel, filename, cmap='viridis', logscale=True):
    """
    2Dヒストグラムをプロットする関数。
    対数スケール(LogNorm)にも対応。
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    # 対数スケール指定
    norm = LogNorm(vmin=1, vmax=255) if logscale else None

    # imshowでヒートマップ描画
    im = ax.imshow(hist.T, origin='lower', cmap=cmap,
                   extent=[0,1,0,1], aspect='auto', norm=norm)

    ax.set_title(f"{title}: {filename}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # カラーバー表示
    fig.colorbar(im, ax=ax, label='Pixel Count' if logscale else 'Value')
    return fig


# =======================================
# メイン処理関数
# =======================================
def CreateHistogram_rg_gb(image_path, output_path):
    """
    画像を読み込み、rg・gb平面の2Dヒストグラムを生成・保存・可視化する。

    Parameters:
        image_path : str
            入力画像のパス
        output_path : str
            npy出力フォルダのパス
    """

    # 画像読み込み
    filename, img = load_image(image_path)
    # 正規化と有効マスク生成
    _, rgb_normalized , valid_mask = normalize_image(img)

    # ==============================
    # 2D ヒストグラム（rg空間 & gb空間）
    # ==============================
    hist_rg_2d, hist_gb_2d = compute_2d_histograms(rgb_normalized, valid_mask)

    # # 0〜1正規化
    # presence_rg = normalize_histogram(hist_rg_2d)
    # presence_gb = normalize_histogram(hist_gb_2d)

    # rgとgbを結合した224x224画像を作成
    stacked, combined = combine_rg_gb_histograms(hist_rg_2d, hist_gb_2d)

    # Numpy形式で保存
    np.save(os.path.join(output_path, f"{filename}.npy"), stacked)
    print(f"Saved upsampled 224x224x2 histogram to: {filename}.npy")

    # # ==============================
    # # プロット（可視化）
    # # ==============================
    # fig1 = plot_2d_histogram(hist_rg_2d, "2D Hist (rg count)", 'r = R/(R+G+B)', 'g = G/(R+G+B)', filename)
    # fig2 = plot_2d_histogram(hist_gb_2d, "2D Hist (gb count)", 'g = G/(R+G+B)', 'b = B/(R+G+B)', filename)
    # fig3 = plot_2d_histogram(combined, "RG & GB Combined Histogram", '', '', filename, logscale=True)

    # # ==============================
    # # 複数ウィンドウの位置調整
    # # ==============================
    # move_figure(fig1, 0, 20)        # 左端
    # move_figure(fig2, 30, 20)       # 中央寄り
    # move_figure(fig3, 60, 20)       # 右端

    # # ==============================
    # # スペースキーで全ウィンドウを閉じる
    # # ==============================
    # def on_key(event):
    #     if event.key == ' ':
    #         for f in [fig1, fig2, fig3]:
    #             plt.close(f)

    # for f in [fig1, fig2, fig3]:
    #     f.canvas.mpl_connect("key_press_event", on_key)

    # # ======================
    # # 表示ブロック（終了待ち）
    # # ======================
    # plt.show(block=True)
