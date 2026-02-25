import json
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

def load_dataset(npy_dir, json_path):
    # 1. JSON読み込みと辞書化
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON: {e}")

    rgb_dict = {item["filename"]: item["real_rgb"] for item in json_data}

    # 2. 特徴量とラベルの蓄積
    X_list = []
    y_list = []

    for filename in sorted(os.listdir(npy_dir)):
        if not filename.endswith("_masked.npy"):
            continue

        base_id = filename.replace("_masked.npy", "")
        if base_id not in rgb_dict:
            logging.warning(f"Warning: {base_id} not in JSON, skipping")
            continue

        npy_path = os.path.join(npy_dir, filename)
        try:
            arr = np.load(npy_path)
        except Exception as e:
            logging.warning(f"Failed to load {npy_path}: {e}")
            continue

        X_list.append(arr)  # shape: (224, 224)

        R, G, B = rgb_dict[base_id]
        y_list.append([R, G, B])

    # numpy配列に変換
    X = np.stack(X_list)  # shape: (N, 224, 224)
    y = np.array(y_list)  # shape: (N, 3)

    # DataFrameで返す
    y_df = pd.DataFrame(y, columns=["R", "G", "B"])

    return X, y_df
