# データ分割

## LCC-v0 / LCC-v5（split_LCC）
分割基準:
  Canon 1Ds / Canon 5D が均等になるように分割

分割比率:
  train : 70%
  val   : 15%
  test  : 15%

分割方法:
  1. カメラごとに分割
  2. 各ドメインを結合

---

## Transfer（split_transfer）

分割基準:
  屋内 / 屋外 / 植物 が均等になるように分割

分割比率:
  train : 70%
  val   : 15%
  test  : 15%

分割方法:
  1. 各カテゴリごとに分割
  2. すべて結合

---
## 実行方法

```bash
python split_LCC.py
python split_transfer.py
```

---