import torch
import numpy as np
import matplotlib.pyplot as plt

from HistogramDataset import HistogramDataset
from ResNetModel import ResNetModel
from load_dataset import load_dataset
from config import OUTPUT_DIR, DEVICE, REAL_RGB_JSON_PATH

# ===============================
# Grad-CAMクラス
# ===============================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # layer4 にhook
        target_layer = self.model.backbone[7]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x):
        self.model.zero_grad()

        output = self.model(x)
        idx = torch.argmax(output[0])
        score = output[0, idx]
        score.backward()

        grads = self.gradients[0]
        fmap = self.features[0]

        weights = grads.mean(dim=(1, 2))

        cam = torch.zeros(fmap.shape[1:], device=fmap.device)
        for i, w in enumerate(weights):
            cam += w * fmap[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        print("CAM stats:", cam.min(), cam.max(), cam.mean())
        return cam.cpu().detach().numpy()

# ===============================
# main
# ===============================
def main():

    # ===== データ1枚だけロード =====
    X_test, _ = load_dataset("E:/ColorConstancy/test", REAL_RGB_JSON_PATH)

    x = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ===== モデル =====
    model = ResNetModel().to(DEVICE)
    model.load_state_dict(torch.load(OUTPUT_DIR / "model1.pth"))
    model.eval()

    gradcam = GradCAM(model)
    cam = gradcam.generate(x)

    # ===== 元npy =====
    input_map = X_test[0]
    if input_map.ndim == 3:
        input_map = input_map.mean(axis=0)

    # ===== 特徴マップ =====
    fmap = gradcam.features[0]
    avg_map = fmap.mean(dim=0)  # 7x7

    vals = avg_map.flatten()

    # 負値を消す
    vals = vals - vals.min()

    # 割合に変換（総和1）
    ratio = vals / (vals.sum() + 1e-8)
    ratio_map = ratio.view(7, 7)

    ratio_np = ratio_map.cpu().detach().numpy()

    # ===== 3枚同時表示 =====
    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.imshow(input_map.T, origin="lower")
    plt.title("Input (npy)")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(ratio_np.T, origin="lower")
    plt.title("7x7 Feature")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(cam.T, origin="lower")
    plt.title("Grad-CAM")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
