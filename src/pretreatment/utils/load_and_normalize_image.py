import os 
import cv2
import numpy as np

def load_and_normalize_image(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    sum_rgb = np.sum(img, axis=2, keepdims=True) + 1e-6
    rgb_ratio = img / sum_rgb

    # L1ノルムで正規化（1行ずつのベクトル）
    l1_norm = np.sum(np.abs(img), axis=2, keepdims=True)
    l1_norm[l1_norm == 0] = 1e-6
    rgb_normalized = img / l1_norm

    black_mask = np.any(img > 0, axis=2)
    valid_mask = black_mask & (sum_rgb[:, :, 0] > 1e-6)

    return filename, img, rgb_ratio, rgb_normalized, valid_mask