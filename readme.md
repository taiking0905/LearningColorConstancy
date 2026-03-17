## 目標

既存の色恒常性モデル（LCC-v5）をiPhone撮影画像に適応させ、
照明色推定精度の向上を検証する。

---

## 先行研究

既存モデルは主に専用カメラデータ（[Gehler-Shi](https://www2.cs.sfu.ca/~colour/data/shi_gehler/)）で学習されている。  
しかしスマートフォン画像はスペクトル感度やイメージセンサーなど、ハードウェア的違いがたくさんある。

---

## 方法

- Base Model: LCC-v5（Buzzelli et al., 2023）
- 入力: rg / gb ヒストグラム
- Backbone: ResNet-18
- 評価指標: Mean Angular Error

### データセット
- ImageNet（初期パラメータとしてのみ採用）
- Gehler-Shi（Canon 1Ds、Canon 5Dの二つの一眼レフ）
- 自作iPhone（屋内:50枚、屋外:50枚、植物:50枚）

### 手法
- Gehler-Shi学習済みモデルを初期値として転移学習
- Von Kries対角補正でホワイトバランス確認

---

## 結果

| Dataset | Baseline | Transfer |
|----------|----------|----------|
| Gehler-Shi | 2.29° | - |
| iPhone | 6.73° | **3.47°** |

---

## 考察

iPhone画像ではスペクトル感度差により精度低下が確認された。
転移学習により改善は見られたが、
完全な吸収にはモデル構造の拡張が必要である可能性がある。

## ディレクトリ構造
### 開発環境
``` bash
LearningColorConstancy/
│
├── LearningColorConstancy_exhibit/   # 研究展示用Webサイト（GitHub Pages）
│
├── src/                              # 研究コード
│   ├── LCC-v0/                       # 1997年の先行研究実装
│   ├── LCC-v5/                       # 2023年 Buzzelli et al. 実装
│   ├── Transfer/                     # iPhone画像への転移学習コード
│   ├── Pretreatment/                 # 前処理（マスク処理・ヒストグラム作成）
│   ├── OneDriveServer/               # RAW画像遠隔変換用サーバー
│   └── Test/                         # 発表用画像生成・分析用コード
│
└── requirements.txt                  # 使用ライブラリ一覧
```
### データセット（外部USB）

⚠ データセットは外部USBに保存されている
（ドライブは D: または E: のどちらかになるため注意）

```　bash
D:/ColorConstancy/
│
├── datasets/               # 前処理前のRAW画像
├── masking/                # マスク処理のみ適用
├── enddatasets/            # 前処理完了画像
│
├── histogram/              # LCC-v0用ヒストグラム
├── histogram_rg_gb/        # LCC-v5用ヒストグラム
│
├── LCC-v0/                 # 学習可能形式のLCC-v0データ
├── LCC-v5/                 # 学習可能形式のLCC-v5データ
├── TransferDataset/        # iPhone転移学習用データ
│
└── colorchecker/           # 切り出したカラーチェッカー情報
```

## インストール

インストール手順はこちらを参照してください。

[インストール　ガイド](docs/install.md)

## コード説明
どんなコードかを説明しているmd

[LCC-v0](src/LCC-v0/LCC-v0.md)  
[LCC-v5](src/LCC-v5/LCC-v5.md)  
[OneDriveServer](src/OneDriveServer/OneDriveServer.md)  
[pretrement](src/pretreatment/Pretreatment.md)  
[split_dataset](src/split_dataset/Split_dataset.md)     
[Test](src/Test/Test.md)    
[transfer](src/transfer/Transfer.md)    

## そのほか説明
全体設定    
[cinfig](docs/config.md)

iPhone設定  
[iPhone](docs/iPhone.md)    

## 展示用 pages

[LCC展示用](https://taiking0905.github.io/LearningColorConstancy_exhibit/)