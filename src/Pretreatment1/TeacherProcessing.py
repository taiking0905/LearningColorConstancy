import matplotlib.pyplot as plt
import cv2
import numpy as np
import json


from config import move_figure
# 教師データを処理する関数
def TeacherProcessing(filename, real_rgb_json, image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    # 教師データ読み込み
    with open(real_rgb_json, "r") as f:
        real_rgb_data = json.load(f)
    
    real_rgb = None
    # 対応するreal_rgbを探す
    for item in real_rgb_data:
        if item["filename"] == filename:
            real_rgb = np.array(item["real_rgb"], dtype=np.float32)
            break
    
    if real_rgb is None:
        raise ValueError(f"{filename} の real_rgb が見つかりません。")
    
    # スケーリング係数計算
    scaling_factors = real_rgb / np.mean(real_rgb)
    
    # スケーリング係数を適用
    img_corrected = img / scaling_factors  # ブロードキャストされる
    
    # クリップして0〜1に正規化 (ここは元画像のスケールに依存)
    img_corrected = np.clip(img_corrected, 0, 255)
    
    # uint8に変換して保存
    img_to_save = img_corrected.astype(np.uint8)
    cv2.imwrite(output_path, img_to_save)
    print(f"Corrected image saved to: {output_path}")
    
    # 表示（matplotlib）→ Enter 押しで閉じる
    fig = plt.subplots(figsize=(12, 9)) 
    move_figure(fig, 0, 0) 
    plt.imshow(cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB))
    plt.title("Corrected Image - Press Enter to close")
    plt.axis("off")

    def on_key(event):
        if event.key == ' ':
            plt.close()

    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()