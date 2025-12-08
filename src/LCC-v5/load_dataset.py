import json
import logging
import pandas as pd
import numpy as np
import os

logging.basicConfig(level=logging.INFO)

def load_dataset_from_list(file_list, json_path=None):
    """
    file_list: npyファイルの絶対パスリスト
    json_path: RGBラベルを持つJSONのパス（必要なら）
    """
    # 1. JSON読み込みと辞書化（必要なら）
    rgb_dict = None
    if json_path is not None:
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            rgb_dict = {item["filename"]: item["real_rgb"] for item in json_data}
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON: {e}")

    X_list = []
    y_list = []

    for npy_path in file_list:
        if not os.path.exists(npy_path):
            logging.warning(f"{npy_path} does not exist, skipping")
            continue
        try:
            arr = np.load(npy_path)
        except Exception as e:
            logging.warning(f"Failed to load {npy_path}: {e}")
            continue

        X_list.append(arr)

        # JSONがある場合はラベル取得
        if rgb_dict is not None:
            filename = os.path.basename(npy_path)
            base_id = filename.replace("_masked.npy", "")
            if base_id not in rgb_dict:
                logging.warning(f"Warning: {base_id} not in JSON, skipping label")
                continue
            R, G, B = rgb_dict[base_id]
            y_list.append([R, G, B])
    
    X = np.stack(X_list)  # shape: (N, 224, 224)
    
    if y_list:
        y_df = pd.DataFrame(y_list, columns=["R", "G", "B"])
    else:
        y_df = None

    return X, y_df
