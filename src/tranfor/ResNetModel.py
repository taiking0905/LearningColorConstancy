import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from config import DROPOUT, OUTPUT_DIM, DEVICE

# ResNet18をベースにしたカラー推定モデル（入力: 1ch, 出力: RGBベクトル）
class ResNetModel(nn.Module):
    def __init__(self, output_dim = OUTPUT_DIM, dropout_rate = DROPOUT):
        super().__init__()
        model = resnet18(weights=None)

        # 入力チャンネルを1ch（グレースケールやヒストグラム画像）に変更
        model.conv1 = nn.Conv2d(
            in_channels=1,       # 入力は1チャネル
            out_channels=64,     # 出力は64チャネル（ResNet標準）
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Dropout層を中間に挿入（例: layer3 の後に）
        model.layer3 = nn.Sequential(
            model.layer3,
            nn.Dropout(p=dropout_rate)  # 👈 中間Dropout追加！
        )

        # 最後のfc層にもDropout
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, output_dim)
        )
        self.model = model
        
    def forward(self, x):
        return self.model(x)  # 順伝播（出力は shape: [batch_size, 3]）


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
