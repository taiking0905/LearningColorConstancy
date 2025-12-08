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
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        nn.init.kaiming_normal_(base.conv1.weight, mode='fan_out', nonlinearity='relu')

        # ---- layer3 の後に Dropout 追加 ----
        base.layer3 = nn.Sequential(
            base.layer3,
            nn.Dropout(p=dropout_rate)
        )

        # ---- FC を強化 ----
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim)
        )

        self.backbone = base   # 名前を分けておく

    def forward(self, x):

        # ---- ResNet18, layer4 まで ----
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # ---- Global average pooling ----
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # ---- FC ----
        illum = self.backbone.fc(x)

        # ---- L2 normalize (角度誤差用) ----
        illum = illum / (illum.norm(dim=1, keepdim=True) + 1e-8)

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
