import cv2
import numpy as np

img = cv2.imread(r"E:\ColorConstancy\enddatasets\8D5U5524.png", cv2.IMREAD_UNCHANGED) # 16bit PNGならこのまま
img = img.astype(np.float32)

Rw, Gw, Bw = 769, 1043, 653  # 例: 12bit値

kr, kg, kb = 4095/Rw, 4095/Gw, 4095/Bw  # 12bitの最大値で揃える場合

img_corrected = img.copy()
img_corrected[...,0] *= kr  # B
img_corrected[...,1] *= kg  # G
img_corrected[...,2] *= kb  # R

# 12bit範囲にクリップ
img_corrected = np.clip(img_corrected, 0, 4095).astype(np.uint16)

cv2.imwrite(r"E:\ColorConstancy\test.png", img_corrected)
