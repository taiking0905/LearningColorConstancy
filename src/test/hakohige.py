import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 例：条件ごとの要約統計（あなたの値に置き換えてください）
data_stats = [
    {'label':'Canon EOS 5D', 'q1':1.7986 , 'median':5.6770 , 'q3':10.0602 , 'mean':5.5028, 'p95':13.1573 , 'p99':13.1573, 'max':13.7675},
    {'label':'Canon EOS 1Ds', 'q1':0.3316, 'median':1.1066, 'q3':4.8869, 'mean':1.9726, 'p95':5.8898, 'p99':9.5330, 'max':10.1975},
    {'label':'iPhone', 'q1':3.4857, 'median':5.9504, 'q3':11.9477, 'mean':6.7348, 'p95':13.5234, 'p99':14.2197, 'max':14.3375}
]

fig, ax = plt.subplots(figsize=(8,5))

for i, s in enumerate(data_stats):
    x = i + 1
    q1, q3, med = s['q1'], s['q3'], s['median']
    iqr = q3 - q1
    # 標準的なひげ（近似）:
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    # ただし実データが無いので、上ひげは利用可能な p95/p99/max のうち最小のものを選ぶのも手
    # ここでは upper_whisker を p95 と比較して小さい方を使う（外れ値を抑える処理）
    upper_candidates = [v for v in [s.get('p95'), s.get('p99'), s.get('max'), upper_whisker] if v is not None]
    upper_whisker_final = min(upper_candidates)

    # 箱
    ax.add_patch(Rectangle((x-0.2, q1), 0.4, q3-q1, fill=False))
    # 中央線
    ax.plot([x-0.2, x+0.2], [med, med], color = 'black', linewidth=3)
    # ひげ（上のみ）
    ax.plot([x, x], [q3, upper_whisker_final], color = 'black', linestyle='-')
    # 上端の小横線
    ax.plot([x-0.08, x+0.08], [upper_whisker_final,  upper_whisker_final], color = 'black')
    # 平均プロット
    ax.scatter(x, s['mean'], marker='D', s=50, edgecolor='black', color='orange', zorder=10)

ax.axhline(5, color='blue', linestyle='--', linewidth=2)
ax.axhline(2, color='red', linestyle='--', linewidth=2)

# 軸・ラベル
ax.set_xticks([1,2,3])
ax.set_ylabel('Value')
ax.legend()
plt.tight_layout()
plt.show()
