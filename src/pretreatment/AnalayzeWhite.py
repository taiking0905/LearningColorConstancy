import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from config import move_figure
from image_util import to_8bit_gamma

def save_color_to_json(filename, mean_color, json_path):
    # 画像ファイル名から拡張子と "_masked" を取り除く
    basename = os.path.basename(filename)
    name = basename.replace('_checker.png', '')

    # 辞書として整形
    entry = {
        "filename": name,
        "real_rgb": [float(c) for c in mean_color]
    }

    # すでにファイルが存在していれば読み込む、なければ新規作成
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    # 同じファイル名のデータが既にあれば上書き、なければ追加
    data = [e for e in data if e["filename"] != name]
    data.append(entry)

    # JSONとして保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSONに保存しました: {name}")

def analyze_white_patch(image_checker_path, real_rgb_json):
    # 画像の読み込み（補正前前提）
    img = cv2.imread(image_checker_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 表示用（γ補正 + 拡大）
    display_img = to_8bit_gamma(img)
    display_img = cv2.resize(display_img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
   
    coords = []
    

    def onclick(event):
        if event.xdata and event.ydata and len(coords) < 4:
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            ax.plot(x, y, 'go')
            ax.text(x+5, y-5, f"{len(coords)}", color="green", fontsize=12)
            fig.canvas.draw()
            if len(coords) == 4:
                ax.set_title("PLEASE PRESS 'ENTER' TO CONTINUE")
                fig.canvas.draw()

    def onkey(event):
        if event.key == 'r':
            coords.clear()
            ax.clear()
            ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            ax.set_title("Click 4 points (Press 'r' to reset)")
            fig.canvas.draw()
        elif event.key == ' ':
            plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    move_figure(fig, 0, 20)
    ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Click 4 points for white patch region")
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    if len(coords) < 4:
        print("❌ 4点選択されませんでした。中止します。")
        return None

    # 座標をリサイズ前に戻す
    coords = np.array(coords) / 4.0
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

    # 選択領域のピクセルを取得
    region_pixels = img[mask == 1].reshape(-1, 3)
    colors, counts = np.unique(region_pixels, axis=0, return_counts=True)

    # 最大出現数の色を基準に
    max_idx = counts.argmax()
    base_color = colors[max_idx]  # shape: (3,)

    # ユークリッド距離を計算
    dists = np.linalg.norm(colors - base_color, axis=1)

    # 閾値内の色をフィルタ
    distance_threshold = 15  # 例: 距離10以内の色だけ使う
    mask_valid = dists <= distance_threshold
    filtered_colors = colors[mask_valid]
    filtered_counts = counts[mask_valid]

    # 採用された色があるか確認
    if len(filtered_colors) == 0:
        print("❌ 有効な色が存在しません（距離条件を満たす色なし）")
        return None

    # 採用された色とピクセル数を表示
    # print("✅ 採用された色とそのピクセル数（ユークリッド距離条件を満たすもの）:")
    # for i, (color, count) in enumerate(zip(filtered_colors, filtered_counts)):
    #     print(f"{i+1:2d}. 色 (R,G,B): {color} → ピクセル数: {count}")

    # 平均色（重み付き）の計算と表示
    mean_color = (filtered_colors * filtered_counts[:, None]).sum(axis=0) / filtered_counts.sum()
    print("\n🎯 平均色 (R,G,B):", mean_color.astype(int))
    # BGR→RGBに変換して保存
    save_color_to_json(image_checker_path, mean_color[::-1], real_rgb_json)

