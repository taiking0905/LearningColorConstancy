import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 例：条件ごとの要約統計（あなたの値に置き換えてください）
data_stats = [
    {'label':'Gehler (Orginal)', 'q1':0.33 , 'median':1.24 , 'q3':5.39 , 'mean':2.12, 'p95':7.14 , 'p99':12.19, 'max':16.52},
    {'label':'Gehler (Reproduced)', 'q1':0.8890, 'median':1.8897, 'q3':4.6283, 'mean':2.3163, 'p95':6.1509, 'p99':9.0583, 'max':9.1319},
    {'label':'iPhone RAW', 'q1':4.5362, 'median':7.5115, 'q3':16.6271, 'mean':9.0420, 'p95':18.2456, 'p99':18.9819, 'max':19.1659},
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
    ax.scatter(x, s['mean'], marker='D', s=50, edgecolor='black', color='orange', zorder=10, label='Mean' if i==0 else "")

ax.axhline(5, color='blue', linestyle='--', linewidth=2, label='GrayWorld(5°)')
ax.axhline(2, color='red', linestyle='--', linewidth=2, label='MachineLearning(2°)')

# 軸・ラベル
ax.set_xticks([1,2,3])
ax.set_xticklabels([s['label'] for s in data_stats])
ax.set_ylabel('Value')
ax.legend()
plt.tight_layout()
plt.show()
