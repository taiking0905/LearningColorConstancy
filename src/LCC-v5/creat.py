import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def correct_image_color(image_rgb_normalized, illumination_vector):
    """
    illumination_vector: [R, G, B]（照明色）
    """
    illumination_vector = normalize_vector(illumination_vector)
    gain = np.mean(illumination_vector) / illumination_vector
    corrected = image_rgb_normalized * gain
    return np.clip(corrected, 0, 1)


def gamma_correction(img, gamma=2.2, scale=1):
    """
    可視化用ガンマ補正 + 輝度スケーリング
    img   : [0,1] に正規化された画像
    gamma : ガンマ値（小さいほど明るくなる）
    scale : ガンマ後のスケーリング係数
    """
    img_gamma = np.power(img, 1.0 / gamma)
    img_scaled = scale * img_gamma
    return np.clip(img_scaled, 0, 1)

def to_uint8(img_float):
    return (img_float * 255).astype(np.uint8)

def main(png_path: Path):
    # 12bit PNG 読み込み（16bitとして読まれる）
    image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(png_path)

    # BGR → RGB / 0–1 正規化
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 4095.0

    # 照明色（手動）
    # 転移前
    # illumination = np.array([0.2250, 0.4536, 0.3213], dtype=np.float32)
    # 転移後
    # illumination = np.array([0.2353, 0.4684, 0.2962], dtype=np.float32)
    # 正解
    illumination = np.array([0.2505, 0.5284, 0.2211], dtype=np.float32)

    corrected = correct_image_color(image_rgb, illumination)
    corrected = gamma_correction(corrected)

    # =========================
    # ★ 保存を追加
    # =========================
    save_path =  f"{png_path.stem}_illum_corrected.png"

    corrected_uint8 = to_uint8(corrected)
    corrected_bgr = cv2.cvtColor(corrected_uint8, cv2.COLOR_RGB2BGR)
    pre = cv2.cvtColor(to_uint8(gamma_correction(image_rgb)), cv2.COLOR_RGB2BGR)
    cv2.imwrite( f"{png_path.stem}_illum_corrected.png", corrected_bgr)
    cv2.imwrite( f"{png_path.stem}.png", pre)

    print(f"✅ 保存しました: {save_path}")

    # 表示
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(to_uint8(gamma_correction(image_rgb)))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Illumination Corrected")
    plt.imshow(corrected_uint8)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    png_file = Path("D:\ColorConstancy\masking\Gehler'sRawDataset\8D5U5524_masked.png")
    main(png_file)
