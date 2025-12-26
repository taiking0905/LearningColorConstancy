import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from config import DROPOUT, OUTPUT_DIM, DEVICE

# ResNet18をベースにしたカラー推定モデル
class ResNetModel(nn.Module):
    def __init__(self, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT):
        super().__init__()
        
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ---- conv1 置き換え ----
        base.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=1,
            bias=False
        )
        base.bn1.reset_parameters()
        nn.init.kaiming_normal_(base.conv1.weight, mode='fan_out', nonlinearity='relu')

        # ---- FC を強化 ----
        self.backbone = nn.Sequential(
            *list(base.children())[:-1]  # layer4 まで + avgpool まで
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)        # layer4 + avgpoolまで
        x = torch.flatten(x, 1)
        x = self.dropout(x)         # 過学習防止
        illum = self.fc(x)
        illum = illum / (illum.norm(dim=1, keepdim=True) + 1e-8)  # L2 normalize
        return illum


# 🔺角度ベースの損失関数（色ベクトルの方向を比較）
def angular_loss(pred, target):
    """
    pred: モデルの出力 (N, 3)
    target: 正解のRGB比率ベクトル (N, 3)
    → 出力ベクトルと正解ベクトルの角度（cos類似度）で誤差を計算
    """
    pred_norm = F.normalize(pred, dim=1)     # 出力をL2正規化（長さを1に）
    target_norm = F.normalize(target, dim=1) # 正解もL2正規化

    cos_sim = (pred_norm * target_norm).sum(dim=1)  # 各ベクトル間のcos類似度
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    loss = 1 - cos_sim  # cosθが高い（方向が一致）ほど損失が小さい
    return loss.mean()  # バッチ平均の損失を返す

def compute_angular_errors(y_pred_all, y_true_all):
    y_pred_norm = y_pred_all / np.linalg.norm(y_pred_all, axis=1, keepdims=True)
    y_true_norm = y_true_all / np.linalg.norm(y_true_all, axis=1, keepdims=True)
    dot_products = np.clip(np.sum(y_pred_norm * y_true_norm, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot_products))

# 🔁 1エポック分の訓練処理
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()

    angular_errors_all = []
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---- 角度誤差（評価用）----
        with torch.no_grad():
            pred_np = pred.detach().cpu().numpy()
            y_np = y_batch.detach().cpu().numpy()
            ang = compute_angular_errors(pred_np, y_np)
            angular_errors_all.extend(ang)

    avg_loss = total_loss / len(loader)
    mean_ang = np.mean(angular_errors_all)

    return avg_loss, mean_ang, np.array(angular_errors_all)

# 🔍 評価関数（検証・テスト用）
def evaluate(model, loader, loss_fn):
    model.eval()

    angular_errors_all = []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()

            pred_np = pred.cpu().numpy()
            y_np = y_batch.cpu().numpy()
            ang = compute_angular_errors(pred_np, y_np)
            angular_errors_all.extend(ang)

    avg_loss = total_loss / len(loader)
    mean_ang = np.mean(angular_errors_all)

    return avg_loss, mean_ang, np.array(angular_errors_all)
