import json
import numpy as np

# JSONファイル読み込み
with open("D:/ColorConstancy/real_rgb.json", "r") as f:
    data = json.load(f)

# デバイスごとのRGBリスト
devices = {
    "8D": {"R": [], "G": [], "B": []},
    "IMG_0": {"R": [], "G": [], "B": []},
    "IMG_4": {"R": [], "G": [], "B": []},
}

# デバイス分類と集計
for entry in data:
    filename = entry["filename"]
    rgb = entry["real_rgb"]
    
    if filename.startswith("8D"):
        dev = "8D"
    elif filename.startswith("IMG_0"):
        dev = "IMG_0"
    elif filename.startswith("IMG_4"):
        dev = "IMG_4"
    else:
        continue

    # 正規化（明るさで割る）
    total = sum(rgb)
    if total == 0:
        norm_rgb = [0, 0, 0]
    else:
        norm_rgb = [c / total for c in rgb]

    devices[dev]["R"].append(norm_rgb[0])
    devices[dev]["G"].append(norm_rgb[1])
    devices[dev]["B"].append(norm_rgb[2])

# 平均・分散・CV計算関数（丸めなし）
def stats_rgb(arr_r, arr_g, arr_b):
    R_arr = np.array(arr_r)
    G_arr = np.array(arr_g)
    B_arr = np.array(arr_b)
    
    # 平均
    R_avg, G_avg, B_avg = R_arr.mean(), G_arr.mean(), B_arr.mean()
    
    # 分散
    R_var, G_var, B_var = R_arr.var(), G_arr.var(), B_arr.var()
    
    # CV（標準偏差 / 平均）
    R_cv = np.sqrt(R_var) / R_avg if R_avg != 0 else 0
    G_cv = np.sqrt(G_var) / G_avg if G_avg != 0 else 0
    B_cv = np.sqrt(B_var) / B_avg if B_avg != 0 else 0
    
    return (R_avg, G_avg, B_avg), (R_var, G_var, B_var), (R_cv, G_cv, B_cv)

# 出力
for dev, vals in devices.items():
    if len(vals["R"]) == 0:
        print(f"{dev}: データなし\n")
        continue
    
    avg_rgb, var_rgb, cv_rgb = stats_rgb(vals["R"], vals["G"], vals["B"])
    
    print(f"{dev} 枚数: {len(vals['R'])}")
    print(f"平均RGB比率: r={avg_rgb[0]}, g={avg_rgb[1]}, b={avg_rgb[2]}")
    print(f"分散: r={var_rgb[0]}, g={var_rgb[1]}, b={var_rgb[2]}")
    print(f"CV: r={cv_rgb[0]}, g={cv_rgb[1]}, b={cv_rgb[2]}\n")
