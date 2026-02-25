import numpy as np
# data = np.load("C:/Users/taiki/Desktop/LCC-v5消さないように/test/8D5U5524_masked.npy")
# data = np.load("E:/ColorConstancy/LCC-v5/train/8D5U5524_masked.npy")
data = np.load("E:/ColorConstancy/histogram_rg_gb/8D5U5524_masked.npy")
data = np.squeeze(data)        # shape: (224,224)

print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Min / Max:", data.min(), data.max())
print("Mean / Median:", data.mean(), np.median(data))
print("Unique values:", np.unique(data))

# 簡単に可視化
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# 0 をマスク
masked_zero = np.ma.masked_where(data != 0, data)

# 通常データ（0以外）
plt.imshow(data, cmap='hot')

# 0 の部分だけを上書き表示
plt.imshow(masked_zero, cmap=colors.ListedColormap(['cyan']), alpha=1.0)

plt.colorbar()
plt.show()

num_zeros = np.sum(data == 0)
print("0 の個数:", num_zeros)
# 低値をより細かく区切ったビン
bins = [
    0 , 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, data.max()+1
]

counts, edges = np.histogram(data, bins=bins)

# 結果表示
for i in range(len(counts)):
    print(f"{edges[i]:.7f} ~ {edges[i+1]:.7f} : {counts[i]} 個")

# 可視化
plt.bar(range(len(counts)), counts, tick_label=[f"{edges[i]:.7f}-{edges[i+1]:.7f}" for i in range(len(counts))])
plt.xticks(rotation=45)
plt.ylabel("個数")
plt.title("データ分布ヒストグラム（極小値まで）")
plt.show()