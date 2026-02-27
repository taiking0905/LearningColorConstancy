# Transfer (LCC-v5 Based)

## 概要

Transfer は LCC-v5 を基盤とした転移学習モデルである。  
ImageNet 事前学習済み ResNet-18 を初期重みとして使用し、
iPhone で撮影した独自データセットに適応させることを目的とする。

本モデルでは、層の凍結（Freeze）戦略を切り替えることで、
特徴抽出部分を保持しつつ最適化を行う。

---

## ディレクトリ構造

``` bash
Transfer
├── outputs
│   ├── transfer_model.pth        # 転移学習後のモデル
|   └── resnet18_model.pth        # 転移学習前のモデル
├── config.py                     # 設定ファイル（パス・ハイパーパラメータ）
├── freeze_layers.py              # 凍結設定ファイル
├── load_dataset.py               # データセット読み込み
├── ResNetModel.py                # ResNet-18 カスタムモデル
├── train.py                      # 学習スクリプト
└── test.py                       # 評価スクリプト
```

---

## データセット構造

``` bash
ColorConstancy
├── TransferDataset
│   ├── train
│   ├── val
│   ├── test
│   └── pretest
└── real_rgb.json               # 各画像に対応する正解照明色（R, G, B）
```

---

## 入力 / 出力

Input:
- rg histogram データとgb histogram データを一つにした画像

Output:
- 推定照明色 (R, G, B)

---

## モデル構造

- LCC-v5と同様

---

## 凍結箇所

`FREEZE_MODE` により学習対象層を制御する。

- `none` : 全層学習
- `fc_only` : 全結合層のみ学習
- `layer4_fc` : layer4 と FC を学習
- `deep_blocks` : layer3 以降を学習
- `freeze_only_layer` : conv1 と FC のみ学習

今回は`none`を使用
---

## 学習設定

Loss: AngularLoss  
Optimizer: Adam  
Epochs: 100  
Batch size: 16  
Learning rate: 1e-6  
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

- 学習済みモデル: `outputs/transfer_model.pth`