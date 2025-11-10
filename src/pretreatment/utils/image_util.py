import os 
import cv2
import numpy as np
from config import to_8bit_gamma
from config import BLACK_LEVEL, WHITE_LEVEL

# def load_and_normalize_image(image_path):
#     filename = os.path.splitext(os.path.basename(image_path))[0]
#     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

#     sum_rgb = np.sum(img, axis=2, keepdims=True) + 1e-6
#     rgb_ratio = img / sum_rgb

#     # L1ノルムで正規化（1行ずつのベクトル）
#     l1_norm = np.sum(np.abs(img), axis=2, keepdims=True)
#     l1_norm[l1_norm == 0] = 1e-6
#     rgb_normalized = img / l1_norm

#     black_mask = np.any(img > 0, axis=2)
#     valid_mask = black_mask & (sum_rgb[:, :, 0] > 1e-6)

    
#     # ガンマ補正・8bit変換（表示用）
#     display_img = to_8bit_gamma(img)

#     return filename, img, display_img, rgb_ratio, rgb_normalized, valid_mask

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    return filename, img

def normalize_image(img):
    sum_rgb = np.sum(img, axis=2, keepdims=True) + 1e-6
    rgb_ratio = img / sum_rgb
    l1_norm = np.sum(np.abs(img), axis=2, keepdims=True)
    l1_norm[l1_norm == 0] = 1e-6
    rgb_normalized = img / l1_norm
    valid_mask = (np.any(img > 0, axis=2)) & (sum_rgb[:, :, 0] > 1e-6)
    return rgb_ratio, rgb_normalized, valid_mask

def to_8bit_gamma(img, gamma=3):
    """
    12bitまたは16bit画像を8bitに変換して、ガンマ補正も適用（表示用）
    """
    # 正規化（0〜1）
    img = np.clip((img)/ (WHITE_LEVEL - BLACK_LEVEL), 0, 1)

    # ガンマ補正（sRGB風）
    img_gamma = np.power(img, 1 / gamma)

    # 8bit化
    return (img_gamma * 255).astype(np.uint8)