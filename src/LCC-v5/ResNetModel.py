import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

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


def mixed_loss(pred, target, alpha=0.7):
    # L2 正規化
    pred_n = F.normalize(pred, dim=1)
    target_n = F.normalize(target, dim=1)

    # Angular loss
    cos_sim = (pred_n * target_n).sum(dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angular = 1 - cos_sim

    # L2 loss
    l2 = F.mse_loss(pred_n, target_n, reduction='none').sum(dim=1)

    # 混合
    loss = alpha * angular + (1 - alpha) * l2
    return loss.mean()


# 🔁 1エポック分の訓練処理

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    batch_losses = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        batch_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss, batch_losses

# 🔍 評価関数（検証・テスト用）
def evaluate(model, loader, loss_fn):
    model.eval()  
    total_loss = 0.0
    batch_losses = []

    with torch.no_grad():  # 勾配を計算しない（推論のみで高速・省メモリ）

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            pred = model(X_batch)               # モデル出力
            loss = loss_fn(pred, y_batch)       # 損失を計算
            total_loss += loss.item()
            batch_losses.append(loss.item())

    average_loss = total_loss / len(loader)     # 全体の平均損失
    return average_loss, batch_losses
