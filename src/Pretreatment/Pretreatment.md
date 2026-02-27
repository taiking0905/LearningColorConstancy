# 前処理

## 入力データについて

本研究で使用する「RAW画像」とは、以下の条件を満たす画像を指す
- 12bit センサーデータ
- ガンマ補正なし（Linear RGB）
- カメラ色空間（未sRGB変換）
- ブラックレベル未補正 or 手動補正済み
- 軽量デモザイク処理のみ適用（OneDriveServer参照）

---
## 前処理の流れ（Pretreatment.py）

1. 生画像読み込み   
2. マスク処理（MaskProcessing.py）  
3. ホワイト領域抽出（AnalayzeWhite.py） 
4. ヒストグラム生成     
   ├── LCC-v0 : CreateHistogram.py  
   └── LCC-v5 : CreateHistogram_rg_gb.py    
5. 特徴量保存   
6. 教師データとデータを最終確認（3300.py）  
---

## 前処理実行
``` bash
python Pretreatment.py
```
## 操作方法

q       : 終了
r       : リセット
space   : 次の画像へ

### マスク処理
カラーチェッカーまたはグレーカードを
4隅で囲む（クリック順は自由、クロスしなければOK）

### ホワイト領域抽出
最も明るい白を4隅で囲む（クリック順は自由、クロスしなければOK）

### 注意
白の値が 3300 を超えた場合：
→ r でリセット
→ 次に明るい白を選択
→ 再実行

---
## カメラ設定（black/whiteレベル）をデバイスごとに変更して下さい
BLACK_LEVEL: 0  
WHITE_LEVEL: 4095

---

## ディレクトリ構造

``` bash
Pretreatment
├── 3300.py                     # 白飛び検知（教師データとマスク後の確認）
├── AnalayzeWhite.py            # ホワイト領域解析
├── config.py                   # 前処理用設定ファイル
├── CreateHistogram.py          # ヒストグラム生成（LCC-v0用）
├── CreateHistogram_rg_gb.py    # rg / gb ヒストグラム生成（LCC-v5用）
├── image_util.py               # 画像ユーティリティ関数
├── MaskProcessing.py           # マスク処理
└── Pretreatment.py             # 前処理メインスクリプト
``` 
---
### データセット（外部USB）

```　bash
D:/ColorConstancy/
│
├── datasets/               # 前処理前のRAW画像（ここに前処理したいデータを入れる）
├── masking/                # マスク処理のみ適用（成果物1）
├── enddatasets/            # 前処理完了画像（datasets移動先）
│
├── histogram/              # LCC-v0用ヒストグラム（成果物2）
├── histogram_rg_gb/        # LCC-v5用ヒストグラム（成果物3）
│
└── colorchecker/           # 切り出したカラーチェッカー情報（成果物4）
```