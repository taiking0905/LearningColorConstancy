# LCC-v5

## 概要

LCC-v5 は、CNN を用いた色恒常性推定モデルである。  
RGB画像を直接入力とし、ResNet-18 をバックボーンとして照明色を推定する。

本実装では torchvision に実装された ResNet-18 を利用し、
最終全結合層を照明色推定用（3出力）に置き換えている。  
初期パラメータには ImageNet 事前学習重みを使用している。

---

## ディレクトリ構造

``` bash
LCC-v5
├── outputs
│   └── resnet18_model.pth      # 学習したモデル
├── config.py                   # 設定ファイル
├── create.py                   # 色恒常性性画像表示
├── grad_cam.py                 # Grad-CAM 可視化
├── HistogramDataset.py         # データセット作成
├── load_dataset.py             # データセット読み込み
├── ResNetModel.py              # ResNet-18 のカスタムモデル定義
├── train.py                    # 学習スクリプト
└── test.py                     # 評価スクリプト
```
---

## データセット構造

``` bash
ColorConstancy
├── LCC-v5
│   ├── train
│   ├── val
│   └── test
└── real_rgb.json               # 各画像に対応する正解照明色（R, G, B）
```
---

## Input / Output

Input:
- RGB画像（前処理済み）

Output:
- 推定照明色 (R, G, B)

---

## モデル構造

- Backbone: ResNet-18
- 最終層を 3出力 (R, G, B) に変更
- Global Average Pooling 後に全結合層

---

## 学習設定

Loss: AngularLoss  
Optimizer: Adam  
Epochs: 1000  
Batch size: 16  
Learning rate: 5e-5  
Weight decay: 5e-5  
Dropout: 0.2  

### データ拡張（過学習防止）

Random Erasing:
- ERASE_PROB: 0.3  
- ERASE_SIZE: 30  

---

## 実行方法

```bash
python train.py
python test.py
```

---

## 出力

- 学習済みモデル: `outputs/resnet18_model.pth`