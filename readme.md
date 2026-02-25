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

# LearningColorConstancy_exhibit
# src
## LCC-v0
## LLC-v5
## OneDriveServer
## 
##
# requirements.txt