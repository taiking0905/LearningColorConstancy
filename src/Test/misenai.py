import numpy as np
import cv2
import os

BLACK_LEVEL = 0
WHITE_LEVEL = 4095  # PNG確認用なら255に

def to_8bit_gamma(img, gamma=1):
    img = np.clip((img - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL), 0, 1)
    img_gamma = np.power(img, 1 / gamma)
    return (img_gamma * 255).astype(np.uint8)

def png_to_jpeg(png_path, jpeg_path=None, quality=95):
    # OpenCVで読み込み（16bitもそのまま読める）
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {png_path}")

    # RGB順に変換（PNGは大体RGBだけど保険でBGR→RGB）
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 8bit + ガンマ補正
    rgb_gamma = to_8bit_gamma(img, gamma=2.2)

    if jpeg_path is None:
        jpeg_path = os.path.splitext(png_path)[0] + ".jpeg"

    success = cv2.imwrite(jpeg_path, rgb_gamma, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if success:
        print(f"JPEG保存成功: {jpeg_path}")
    else:
        print(f"JPEG保存失敗: {jpeg_path}")
    return jpeg_path

# 使用例
png_file = r"C:\Users\taiki\OneDrive_SuwaTokyoUniversityOfScience\LearningColorConstancy_IPhone\LearningColorConstancy_rawpng\IMG_4027.png"
jpeg_file = r"E:\IMG_4027.jpeg"
png_to_jpeg(png_file, jpeg_file, quality=95)
