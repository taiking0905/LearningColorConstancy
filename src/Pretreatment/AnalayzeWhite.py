import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import move_figure
from image_util import to_8bit_gamma

# ==========================
# JSON操作用関数
# ==========================
def load_json(json_path):
    """JSONファイルを読み込む。存在しなければ空リストを返す"""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(data, json_path):
    """データをJSONファイルとして保存"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_color_to_json(filename, mean_color, json_path):
    """平均色をJSONに保存"""
    basename = os.path.basename(filename)
    name = basename.replace('_checker.png', '')

    data = load_json(json_path)
    # 既存の同名データを削除して更新
    data = [e for e in data if e["filename"] != name]
    data.append({
        "filename": name,
        "real_rgb": [float(c) for c in mean_color]
    })
    save_json(data, json_path)
    print(f"✅ JSONに保存しました: {name}")

# ==========================
# GUIで白パッチ選択クラス
# ==========================
class WhitePatchSelector:
    def __init__(self, image_path, scale=4):
        """
        画像を読み込み、GUI表示用に拡大
        :param image_path: 入力画像パス
        :param scale: 表示拡大倍率
        """
        self.image_path = image_path
        self.scale = scale
        self.coords = []  # クリック座標を格納
        self.img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        self.display_img = cv2.resize(
            to_8bit_gamma(self.img), 
            None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )

    def onclick(self, event):
        """クリックイベントで座標を取得"""
        if event.xdata and event.ydata and len(self.coords) < 4:
            x, y = int(event.xdata), int(event.ydata)
            self.coords.append((x, y))
            self.ax.plot(x, y, 'go')
            self.ax.text(x+5, y-5, f"{len(self.coords)}", color="green", fontsize=12)
            self.fig.canvas.draw()
            if len(self.coords) == 4:
                self.ax.set_title("PLEASE PRESS 'ENTER' TO CONTINUE")
                self.fig.canvas.draw()

    def onkey(self, event):
        """キーイベント処理: rでリセット、スペースで終了"""
        if event.key == 'r':
            self.coords.clear()
            self.ax.clear()
            self.ax.imshow(cv2.cvtColor(self.display_img, cv2.COLOR_BGR2RGB))
            self.ax.set_title("Click 4 points (Press 'r' to reset)")
            self.fig.canvas.draw()
        elif event.key == ' ':
            plt.close()

    def select_region(self):
        """GUIで白パッチ領域を選択して座標を返す"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        move_figure(self.fig, 0, 20)
        self.ax.imshow(cv2.cvtColor(self.display_img, cv2.COLOR_BGR2RGB))
        self.ax.set_title("Click 4 points for white patch region")
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        plt.show()
        if len(self.coords) < 4:
            print("❌ 4点選択されませんでした。中止します。")
            return None
        # 座標をリサイズ前に戻す
        return np.array(self.coords) / self.scale

# ==========================
# 白パッチ解析関数
# ==========================
def analyze_white_patch(image_checker_path, real_rgb_json):
    """
    白パッチをGUIで選択して平均色を計算し、JSONに保存
    :param image_checker_path: 入力画像パス
    :param real_rgb_json: 保存先JSONパス
    """
    selector = WhitePatchSelector(image_checker_path)
    coords = selector.select_region()
    if coords is None:
        return None

    # 選択領域マスク作成
    mask = np.zeros(selector.img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

    # 選択領域のピクセルを取得
    region_pixels = selector.img[mask == 1].reshape(-1, 3)
    colors, counts = np.unique(region_pixels, axis=0, return_counts=True)

    # 最大出現数の色を基準に
    max_idx = counts.argmax()
    base_color = colors[max_idx]

    # ユークリッド距離で近似色をフィルタ
    distance_threshold = 15
    dists = np.linalg.norm(colors - base_color, axis=1)
    mask_valid = dists <= distance_threshold
    filtered_colors = colors[mask_valid]
    filtered_counts = counts[mask_valid]

    if len(filtered_colors) == 0:
        print("❌ 有効な色が存在しません（距離条件を満たす色なし）")
        return None

    # 平均色を計算（重み付き）
    mean_color = (filtered_colors * filtered_counts[:, None]).sum(axis=0) / filtered_counts.sum()
    print("\n🎯 平均色 (R,G,B):", mean_color.astype(int))

    # BGR→RGBに変換してJSONに保存
    save_color_to_json(image_checker_path, mean_color[::-1], real_rgb_json)
