# LCC-v0

## 概要

LCC-v0 は、1997年に提案された色恒常性手法を再実装したモデルである。  
rgヒストグラム特徴量を入力とし、MLP により照明色を推定する。

---

## ディレクトリ構造

``` bash
LCC-v0
├── outputs
│   └── mlp_model.pth           # 学習したモデル
├── config.py                   # 設定ファイル
├── load_dataset.py             # データセット読み込み
├── MLPModel.py                 # MLP のカスタムモデル定義
├── train.py                    # 学習スクリプト
└── test.py                     # 評価スクリプト
```

---

## データセット構造

``` bash
ColorConstancy
├── LCC-v0
│   ├── train
│   ├── val
│   └── test
└── real_rgb.json               # 各画像に対応する正解照明色（R, G, B）
```

---

## Input / Output

Input:
- rg histogram データ

Output:
- 推定照明色 (R, G)
- B は色正規化条件 R + G + B = 1 に基づき  
  `B = 1 - R - G` として算出

---

## 学習

Loss: MSELoss  
Optimizer: Adam  
Epochs: 500  
Batch size: 16  
Learning rate: 1e-4  
Weight decay: 4e-3  

---

## 実行方法

```bash
python train.py
python test.py
```

---

## 出力

- 学習済みモデル: `outputs/mlp_model.pth`