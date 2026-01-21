import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean
import json

from load_dataset import load_dataset
from HistogramDataset import HistogramDataset
from ResNetModel import ResNetModel, angular_loss, evaluate
from config import BASE_DIR, TEST_DIR, VAL_DIR, REAL_RGB_JSON_PATH, OUTPUT_DIR, BATCH_SIZE, SEED, DEVICE, set_seed

def compute_angular_errors(y_pred_all, y_true_all):
    y_pred_norm = y_pred_all / np.linalg.norm(y_pred_all, axis=1, keepdims=True)
    y_true_norm = y_true_all / np.linalg.norm(y_true_all, axis=1, keepdims=True)
    dot_products = np.clip(np.sum(y_pred_norm * y_true_norm, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot_products))

def visualize_errors(y_pred_all, y_true_all, angular_errors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # (1) ヒストグラム
    plt.figure(figsize=(6, 4))
    plt.hist(angular_errors, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Angular Error (°)")
    plt.ylabel("Count")
    plt.title("Angular Error Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "angular_error_histogram.png")
    plt.close()

    # (2) 箱ひげ図
    plt.figure(figsize=(4, 6))
    plt.boxplot(angular_errors, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen'),
                medianprops=dict(color='red'))
    plt.ylabel("Angular Error (°)")
    plt.title("Angular Error Boxplot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "angular_error_boxplot.png")
    plt.close()

    # (4) サンプルごとの角度誤差
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(len(angular_errors)), angular_errors, c='orange', alpha=0.6)
    plt.axhline(np.mean(angular_errors), color='red', linestyle='--', label=f"Mean = {np.mean(angular_errors):.2f}°")
    plt.xlabel("Sample Index")
    plt.ylabel("Angular Error (°)")
    plt.title("Angular Error per Sample")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "angular_error_scatter.png")
    plt.close()

def tri_mean(data, proportiontocut):
    # 上下25%をカットして平均を計算
    tm = trim_mean(data, proportiontocut)
    return tm

def best_25_percent(data, proportion):
    """最も小さい25%の平均"""
    k = int(len(data) * proportion)
    return np.mean(np.sort(data)[:k])

def worst_25_percent(data, proportion):
    """最も大きい25%の平均"""
    k = int(len(data) * proportion)
    return np.mean(np.sort(data)[-k:])

def main():
    set_seed(SEED)
    base_dir = BASE_DIR
    print("Base dir:", base_dir)
    print("DEVICE:", DEVICE)

    # X_test, y_test_df = load_dataset(TEST_DIR, REAL_RGB_JSON_PATH)
    X_test, y_test_df = load_dataset(VAL_DIR, REAL_RGB_JSON_PATH)
    y_test = y_test_df[["R", "G" , "B"]].values
    val_dataset = HistogramDataset(X_test, y_test)

    # 2. モデルロード
    model = ResNetModel().to(DEVICE)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'model_test.pth'))
    model.eval()

    # 3. 評価
    test_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loss, _ , _= evaluate(model, test_loader, angular_loss)
    print(f"\n📊 Test Loss = {test_loss:.4f}")

    # 4. RGB比較（5件）
    print("\n🎨 Prediction vs Actual (first 5 samples):")
    with torch.no_grad():
        for i in range(min(100, len(X_test))):
            x = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(x)[0].cpu()
            pred /= torch.sum(pred)  # 出力ベクトルを正規化
            pred = pred.unsqueeze(0)

            y_true = torch.tensor(y_test[i], dtype=torch.float32)
            y_true /= torch.sum(y_true)  # 🔧 各y_test[i]を個別に正規化
            y_true = y_true.unsqueeze(0)
            loss = angular_loss(pred, y_true).item()
            cos_sim = 1 - loss

            # print(f"{i+1:2d}: "
            #     f"Pred (r,g,b): ({pred[0,0]:.4f}, {pred[0,1]:.4f}, {pred[0,2]:.4f}) | "
            #     f"True (r,g,b): ({y_true[0,0]:.4f}, {y_true[0,1]:.4f}, {y_true[0,2]:.4f}) | "
            #     f"AngularLoss: {loss:.4f} | CosSim: {cos_sim:.4f}")



    # 5. 可視化と統計
    y_pred_all, y_true_all = [], []
    with torch.no_grad():
        for x, y_true in zip(X_test, y_test):
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(x_tensor)[0].cpu().numpy()
            pred /= np.linalg.norm(pred)
            y_pred_all.append(pred)
            y_true_all.append(y_true)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    angular_errors = compute_angular_errors(y_pred_all, y_true_all)
    visualize_errors(y_pred_all, y_true_all, angular_errors)

    print("\n📏 Angular Error Stats (°):")
    print(f"Method\t\tValue")
    print(f"Mean\t\t{np.mean(angular_errors):.4f}")
    print(f"Median\t\t{np.median(angular_errors):.4f}")
    print(f"Tri-m.\t\t{tri_mean(angular_errors, 0.1):.4f}")
    print(f"B-25\t\t{best_25_percent(angular_errors, 0.25):.4f}")
    print(f"W-25\t\t{worst_25_percent(angular_errors, 0.25):.4f}")
    print(f"95-P\t\t{np.percentile(angular_errors, 95):.4f}")
    print(f"99-P\t\t{np.percentile(angular_errors, 99):.4f}")
    print(f"Max\t\t{np.max(angular_errors):.4f}")


if __name__ == "__main__":
    main()
