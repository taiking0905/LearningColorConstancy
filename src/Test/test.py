import rawpy
import numpy as np
import cv2
import glob
import os
from dotenv import load_dotenv

load_dotenv()
# DNGファイルが入っているディレクトリ
TEST_IPHONE_PATH = os.getenv("TEST_IPHONE_PATH")

# ディレクトリ内のすべてのDNGファイルを取得
dng_files = glob.glob(os.path.join(TEST_IPHONE_PATH, "*.DNG"))

for dng_path in dng_files:
    try:
        with rawpy.imread(dng_path) as raw:
            # WB情報の取得（camera_whitebalance または daylight_whitebalance）
            try:
                as_shot_neutral = np.array(raw.camera_whitebalance, dtype=np.float32)
                if as_shot_neutral is None or np.any(as_shot_neutral == 0):
                    raise ValueError
            except:
                as_shot_neutral = np.array(raw.daylight_whitebalance, dtype=np.float32)

            as_shot_neutral = as_shot_neutral[:3]  # 最初の3要素を使用
            as_shot_neutral[as_shot_neutral == 0] = 1e-6
            wb_multipliers = 1.0 / as_shot_neutral


# これはだめらしい修正町
            # デモザイク後のRGBを16bitで取得（ホワイトバランスなし・ガンマ補正なし）
            rgb = raw.postprocess(
                use_camera_wb=False,
                no_auto_bright=True,
                no_auto_scale=True,
                output_bps=16,
                gamma=(1, 1),
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                #学習用はRAW,評価用はXYZ
                output_color=rawpy.ColorSpace.raw
            )
                    # # ブラックラベル補正も自動でやっている
        # with rawpy.imread(dng_path) as raw:
        #     rgb_raw = raw.postprocess(
        #         # --- 自動補正の排除 ---
        #         use_camera_wb=False,      # カメラWB (AsShotNeutral) を排除
        #         no_auto_bright=True,      # 自動輝度補正を排除
        #         no_auto_scale=True,       # 自動スケール（ダイナミックレンジ補正）を排除
        #         # use_camera_matrix=False,  # カメラの色変換行列 (ColorMatrix) を排除
        #         bright=1.0,               # 輝度乗数を1.0に固定
        #         user_wb=(1.0, 1.0, 1.0, 1.0), # 独自のWBをニュートラルに固定
        #         gamma=(1, 1),             # ガンマ補正を排除 (線形トーン)
        #         user_flip=None,           # 自動回転 (Orientation) を排除
        #         # disable_crop=True,        # デフォルトのクロップを排除
        #         # med_passes=0,             # 軽微なノイズリダクションを排除
        #         # fbdd_no_interpolation=True, # 不良ピクセル補間を排除

        #         # --- 処理品質/形式の維持 ---
        #         output_bps=16,            # 16ビット深度を維持
        #         demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, # デモザイクアルゴリズムを指定
        #         output_color=rawpy.ColorSpace.raw, # カメラのRAW色空間を維持
        #         four_color_rgb=True,      # 4色RGB処理を有効

        #         # --- その他 ---
        #         half_size=False           # デモザイク処理をフル解像度で行うため、half_size=Falseに設定を推奨
        #     )

            # # 保存フォルダ作成
            # os.makedirs(OneDrive_RAW_PNG_PATH, exist_ok=True)
            # os.makedirs(OneDrive_GAMMA_PNG_PATH, exist_ok=True)
            
            # # RAW PNG 保存
            # filename_raw = os.path.splitext(os.path.basename(dng_path))[0] + ".png"
            # save_path_raw = os.path.join(OneDrive_RAW_PNG_PATH, filename_raw)
            # cv2.imwrite(save_path_raw, cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR))
            
            # # 8bit + ガンマ補正
            # rgb_gamma = to_8bit_gamma(rgb_raw)
            # filename_gamma = os.path.splitext(os.path.basename(dng_path))[0] + "_gamma.jpg"
            # save_path_gamma = os.path.join(OneDrive_GAMMA_PNG_PATH, filename_gamma)
            # cv2.imwrite(save_path_gamma, cv2.cvtColor(rgb_gamma, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            # print(f"処理完了: {dng_path}")

            # データの最大値（RAWセンサーの白レベル）を取得
            white_level = raw.white_level
            black_level = raw.black_level_per_channel[0]
            raw_image = raw.raw_image.copy()  
            min_val = raw_image.min()

            # 保存ファイル名（拡張子をPNGに変更）
            filename = os.path.splitext(os.path.basename(dng_path))[0] + ".png"
            save_path = os.path.join(TEST_IPHONE_PATH, filename)

            # PNGとして保存
            cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # ログ出力
            print(f"\nファイル: {dng_path}")
            print("  AsShotNeutral:", as_shot_neutral)
            print("  WB multipliers:", wb_multipliers)
            print("  white_level:", white_level, "black_level:", black_level)
            print("  RAW最小値:", min_val)
            print(f"  保存しました: {save_path}")

    except Exception as e:
        print(f"処理中にエラーが発生しました ({dng_path}): {e}")
