import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from ResNetModel import ResNetModel
from config import DEVICE, OUTPUT_DIR


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def estimate_illumination(model, npy_feature):
    x = torch.tensor(npy_feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()
    return normalize_vector(pred)


def correct_image_color(image_rgb_normalized, illumination_vector):
    illumination_vector = normalize_vector(illumination_vector)
    gain = np.mean(illumination_vector) / illumination_vector
    corrected = image_rgb_normalized * gain
    corrected = np.clip(corrected, 0, 1)
    return corrected


def apply_additional_corrections(image_rgb_normalized, wb_multipliers=None, gamma=2.2):
    """
    ホワイトバランス＋ガンマ補正を行う
    image_rgb_normalized: 0~1 float32
    wb_multipliers: [R, G, B]の乗算係数（Noneなら適用しない）
    """
    img = image_rgb_normalized.copy()
    if wb_multipliers is not None:
        for i in range(3):
            img[..., i] *= wb_multipliers[i]
    img = np.clip(img, 0, 1)
    img = np.power(img, 1.0 / gamma)
    img = np.clip(img, 0, 1)
    return img


def to_uint8(img_float):
    """float32 (0~1) → uint8 (0~255)"""
    return (np.clip(img_float, 0, 1) * 255).astype(np.uint8)


def main(png_path: Path, npy_path: Path):
    # モデルロード
    model = ResNetModel().to(DEVICE)
    state_dict = torch.load(OUTPUT_DIR / 'resnet_model.pth')
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 12bit画像読み込み（→16bitとして読まれる）
    image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"画像が見つかりません: {png_path}")
    if image.dtype != np.uint16:
        raise ValueError("入力画像は16bitである必要があります")

    # RGB変換 + 0〜1に正規化（12bit想定 → 最大値は4095）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb_normalized = image_rgb.astype(np.float32) / 4095.0

    # 特徴量読み込み
    npy_feature = np.load(npy_path)

    # 照明推定
    illumination = estimate_illumination(model, npy_feature)
    print(f"Estimated illumination: {illumination}")

    # 照明推定に基づく補正
    corrected_rgb = correct_image_color(image_rgb_normalized, illumination)

    # AsShotNeutral（例）に基づくWB+ガンマ補正
    as_shot_neutral = np.array([2.562066, 0.9945468, 1.6436796], dtype=np.float32)
    # ここ違う可能性あるぞ
    wb_multipliers = 1.0 / as_shot_neutral

    display_rgb = apply_additional_corrections(image_rgb_normalized, wb_multipliers=None, gamma=2.2)
    corrected_rgb_plus = apply_additional_corrections(corrected_rgb, wb_multipliers=None, gamma=2.2)

    # 保存：float32 → uint8 変換後に保存
    save_path_output = OUTPUT_DIR / f"test_{png_path.name}"
    save_path_illum = OUTPUT_DIR / f"illum_corrected_{png_path.name}"
    save_path_plus = OUTPUT_DIR / f"wb_gamma_corrected_{png_path.name}"

    cv2.imwrite(str(save_path_output), cv2.cvtColor(to_uint8(display_rgb), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_path_illum), cv2.cvtColor(to_uint8(corrected_rgb), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_path_plus), cv2.cvtColor(to_uint8(corrected_rgb_plus), cv2.COLOR_RGB2BGR))

    print(f"✅ 照明推定補正画像を保存しました: {save_path_illum}")
    print(f"✅ WB+ガンマ補正画像を保存しました: {save_path_plus}")

    # 表示（8bit化してから）
    plt.figure(figsize=(12, 6))
    plt.imshow(to_uint8(corrected_rgb_plus))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    png_file = Path(OUTPUT_DIR / "IMG_3891.png")
    npy_file = Path(OUTPUT_DIR / "IMG_3891.npy")
    main(png_file, npy_file)
