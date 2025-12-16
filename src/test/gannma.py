import matplotlib.pyplot as plt
import cv2
import numpy as np

BLACK_LEVEL = 0
WHITE_LEVEL = 4095

def to_8bit_gamma(img, gamma=2.2):
    # 正規化 0~1
    img_norm = np.clip((img - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL), 0, 1)
    # ガンマ補正
    img_gamma = np.power(img_norm, 1/gamma)
    # 8bit化
    return (img_gamma * 255).astype(np.uint8)

# 画像読み込み
# img_path = "C:/Users/taiki/Desktop/8D5U5524_masked.png"
# img_path = 'C:/Users/taiki/Desktop/test/IMG_4017.png'
# img_path = 'C:/Users/taiki/Desktop/test/IMG_4020.png'
# img_path = 'C:/Users/taiki/Desktop/test/DSC01628.png'
img_path = "E:/ColorConstancy/enddatasets/Gehler'sRawDataset/IMG_0861.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
print("読み込み確認:", img.shape, img.dtype, img.min(), img.max())
img = img - BLACK_LEVEL
img = np.clip(img, 0, None)
img = img.astype(np.uint16)
img_display = to_8bit_gamma(img)
img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

# matplotlib で表示
plt.imshow(img_rgb_display)
plt.axis('off')
plt.show()
