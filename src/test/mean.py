import json
import numpy as np

# JSONファイル読み込み
with open("E:/ColorConstancy/real_rgb.json", "r") as f:
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

    devices[dev]["R"].append(rgb[0])
    devices[dev]["G"].append(rgb[1])
    devices[dev]["B"].append(rgb[2])

# 丸め込み・正規化・中央値計算関数
def rounded_normalized_median(arr_r, arr_g, arr_b, decimals=3):
    R_arr = np.array(arr_r)
    G_arr = np.array(arr_g)
    B_arr = np.array(arr_b)
    
    # 中央値
    R_med = np.median(R_arr)
    G_med = np.median(G_arr)
    B_med = np.median(B_arr)
    
    # 正規化して比率
    total_med = R_med + G_med + B_med
    r_ratio_med = round(R_med / total_med, decimals)
    g_ratio_med = round(G_med / total_med, decimals)
    b_ratio_med = round(B_med / total_med, decimals)
    
    return (R_med, G_med, B_med), (r_ratio_med, g_ratio_med, b_ratio_med)

# 平均と比率計算関数
def average_rgb_ratios(arr_r, arr_g, arr_b, decimals=3):
    R_arr = np.array(arr_r)
    G_arr = np.array(arr_g)
    B_arr = np.array(arr_b)
    
    R_avg = R_arr.mean()
    G_avg = G_arr.mean()
    B_avg = B_arr.mean()
    
    total_avg = R_avg + G_avg + B_avg
    r_ratio_avg = round(R_avg / total_avg, decimals)
    g_ratio_avg = round(G_avg / total_avg, decimals)
    b_ratio_avg = round(B_avg / total_avg, decimals)
    
    return (R_avg, G_avg, B_avg), (r_ratio_avg, g_ratio_avg, b_ratio_avg)

# 出力
for dev, vals in devices.items():
    if len(vals["R"]) == 0:
        print(f"{dev}: データなし\n")
        continue
    
    # 平均
    avg_rgb, avg_ratio = average_rgb_ratios(vals["R"], vals["G"], vals["B"])
    # 中央値＋丸め込み
    med_rgb, med_ratio = rounded_normalized_median(vals["R"], vals["G"], vals["B"], decimals=3)
    
    print(f"{dev} 枚数: {len(vals['R'])}")
    print(f"平均RGB比率: r={avg_ratio[0]}, g={avg_ratio[1]}, b={avg_ratio[2]}")
    print(f"中央値RGB比率: r={med_ratio[0]}, g={med_ratio[1]}, b={med_ratio[2]}\n")
