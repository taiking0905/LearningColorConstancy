import torch


from load_dataset import load_dataset
from config import TRAIN_DIR,REAL_RGB_JSON_PATH, DEVICE, set_seed
import psutil
import os



def find_drive_with_folder(folder_name="target_folder"):
    for part in psutil.disk_partitions():
        if 'removable' in part.opts.lower():  # USBなどに限定したいとき
            drive = part.mountpoint
            if os.path.exists(os.path.join(drive, folder_name)):
                return drive
    return None



def main():
    set_seed() 
    
    # 1. データ読み込み
    X_train, y_train_df = load_dataset(TRAIN_DIR, REAL_RGB_JSON_PATH)
    # X_val, y_val_df = load_dataset(VAL_HIST_RG_GB_DIR, REAL_RGB_JSON_PATH)
    # 出力がX= numpy Y=df

    print(X_train.shape)  # → (N, 1, 224, 224)
    print(y_train_df.head())  # → R, G, B列
    print(torch.cuda.is_available())  # TrueならOK
    print(torch.cuda.get_device_name())  # GPU名が出る

    drive = find_drive_with_folder("ColorConstancy")
    if drive:
        print(f"対象のドライブ: {drive}")
    else:
        print("該当フォルダを含むドライブが見つかりませんでした。")


if __name__ == "__main__":
    main()