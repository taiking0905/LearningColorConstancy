import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time 
import numpy as np
import io
from PIL import Image
import logging
from torch.utils.tensorboard import SummaryWriter

from load_dataset import load_dataset
from HistogramDataset import HistogramDataset
from ResNetModel import ResNetModel, angular_loss, train_one_epoch, evaluate
from config import get_base_dir, TRAIN_DIR, VAL_DIR, REAL_RGB_JSON_PATH, EPOCHS, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT, SEED, ERASE_PROB, ERASE_SIZE, DEVICE, set_seed, START_EPOCH_2

def main():
    set_seed(SEED) 
    rng = np.random.default_rng(SEED)
    base_dir = get_base_dir()
    print("Base dir:", base_dir)
    print(torch.cuda.is_available())  # TrueならOK
    print(torch.cuda.get_device_name())  # GPU名が出る

    logging.basicConfig(
    filename=OUTPUT_DIR / 'training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
    writer = SummaryWriter(OUTPUT_DIR / 'tb_logs')

    
    # 1. データ読み込み
    X_train, y_train_df = load_dataset(TRAIN_DIR, REAL_RGB_JSON_PATH)
    X_val, y_val_df = load_dataset(VAL_DIR, REAL_RGB_JSON_PATH)
    # 出力がX= numpy Y=df
    
    # 2. Tensorに変換
    y_train = y_train_df[["R", "G", "B"]].values
    y_val = y_val_df[["R", "G", "B"]].values
    
    print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"X_val.shape = {X_val.shape}, y_val.shape = {y_val.shape}")

    # 3. TensorDataset作成（ここで erase 機能を組み込む）
    train_dataset = HistogramDataset(X_train, y_train, erase_prob=ERASE_PROB, erase_size = ERASE_SIZE,rng=rng)
    val_dataset = HistogramDataset(X_val, y_val,rng=rng)

    # 4. DataLoader作成 pin_memory=Trueこれを使うとGPUへの転送が速くなる
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)


    # 5. モデル定義
    model = ResNetModel().to(DEVICE)
    # model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model_phase1.pth'))
    logging.info("Loaded best_model_phase1.pth for Phase 2 starting.")

    dummy_input = torch.randn(1, 1, 224, 224).to(DEVICE)
    writer.add_graph(model, dummy_input)

    # for name, param in model.named_parameters():
    #     # 'layer1'から'layer4'の勾配計算を無効化（フリーズ）
    #     # ResNetの中間層の重みを保護する
    #     if 'layer' in name:
    #         param.requires_grad = False
    #     # 'conv1'と'fc'層（タスク固有層）は学習させる
    #     else:
    #         param.requires_grad = True
    
    # 2. 全ての層をアンフリーズ
    for param in model.parameters():
        param.requires_grad = True
    logging.info("All layers unfrozen (requires_grad = True).")

    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer_step1 = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT)
    optimizer_step2 = torch.optim.Adam(params, lr=LEARNING_RATE/30, weight_decay=WEIGHT)

    # 損失関数はRGBベクトル間の角度誤差
    loss_fn = angular_loss


    # 学習記録用リスト
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    all_start_time =time.time()

    # Epochループ
    for epoch in range(EPOCHS):
        logging.info(f"==== Epoch {epoch+1}/{EPOCHS} ====")
        epoch_start_time = time.time()
        if(epoch < START_EPOCH_2):
            train_loss, train_batch_losses = train_one_epoch(model, train_loader, optimizer_step1, loss_fn)
        else:
            train_loss, train_batch_losses = train_one_epoch(model, train_loader, optimizer_step2, loss_fn)
        val_loss, val_batch_losses = evaluate(model, val_loader, loss_fn)
        
        epoch_end_time = time.time()
        logging.info(f"Total epoch time: {epoch_end_time - epoch_start_time:.2f} sec")
        logging.info(f"Loss: Train = {train_loss:.4f}, Val = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 検証損失が最小を更新した場合のみモデルを保存
            # ファイル名を 'best_resnet_model.pth' にして、最終保存と区別
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_resnet_model.pth')
            logging.info(f"*** NEW BEST MODEL SAVED! Val Loss: {best_val_loss:.4f} ***")


        # TensorBoard に記録
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_histogram("AngularError/train", np.array(train_batch_losses), epoch+1)
        writer.add_histogram("AngularError/val", np.array(val_batch_losses), epoch+1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    all_end_time = time.time()

    print(f"☆Total all time: {all_end_time - all_start_time:.2f} sec")
    
    # 8. 学習曲線の可視化
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / 'loss_curve.png')
    plt.show()

    writer.close()

if __name__ == "__main__":
    main()