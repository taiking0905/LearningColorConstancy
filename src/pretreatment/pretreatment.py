import os
import glob
import shutil
import logging

from config import setup_directories
from CreateHistogram import CreateHistogram, CreateHistogram_rg_gb
from MaskProcessing import MaskProcessing
from AnalayzeWhite import analyze_white_patch

def process_image(image_path: str, dirs: dict):
    # 画像ファイル名から拡張子を除去して、マスクと教師データのパスを設定
    filename = os.path.splitext(os.path.basename(image_path))[0]

    paths = {
        "masked": os.path.join(dirs["MASK"], f"{filename}_masked.png"),
        "checker": os.path.join(dirs["COLORCHECKER"], f"{filename}_checker.png"),
        "end": os.path.join(dirs["END"], f"{filename}.png"),
    }
    
    # 1. マスク処理
    result = MaskProcessing(image_path, paths["masked"], paths["checker"])
    if result.get("quit"):
        logging.info("Processing stopped by user.")
        return False
    
    # 2. 白色パッチ解析
    analyze_white_patch(paths["checker"], dirs["REAL_RGB_JSON"])

    # 3. ヒストグラム生成
    CreateHistogram(paths["masked"], dirs["HIST"])
    CreateHistogram_rg_gb(paths["masked"], dirs["HIST_RG_GB"])

    # 4. 元画像をENDへ移動
    shutil.move(image_path, paths["end"])
    logging.info(f"Moved {image_path} → {paths['end']}")
    return True

def pretreatment():
    """入力ディレクトリ内の画像に対して一括前処理を実行"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # ディレクトリ設定
    dirs = setup_directories()

    # png画像のパスを取得
    image_paths = sorted(glob.glob(os.path.join(dirs["INPUT"], "*.png")))

    for image_path in image_paths:
        try:
            if not process_image(image_path, dirs):
                break
        except Exception as e:
            logging.exception(f"Error while processing {image_path}: {e}")

# 実行
if __name__ == "__main__":
    pretreatment()
