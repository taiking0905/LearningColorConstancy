import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time 
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from load_dataset import load_dataset
from HistogramDataset import HistogramDataset
from ResNetModel import ResNetModel, mixed_loss, train_one_epoch, evaluate
from config import BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, REAL_RGB_JSON_PATH, EPOCHS, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT, SEED, ERASE_PROB, ERASE_SIZE, DEVICE, set_seed, START_EPOCH_2, START_EPOCH_3

def main():
    set_seed(SEED) 
    rng = np.random.default_rng(SEED)
    base_dir = BASE_DIR
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
    val_dataset = HistogramDataset(X_val, y_val)

    # 4. DataLoader作成 pin_memory=Trueこれを使うとGPUへの転送が速くなる
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)


    # 5. モデル定義
    model = ResNetModel().to(DEVICE)
    print(model)
    # model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model_phase1.pth'))
    logging.info("Loaded best_model_phase1.pth for Phase 2 starting.")

    dummy_input = torch.randn(1, 1, 224, 224).to(DEVICE)
    writer.add_graph(model, dummy_input)

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.08  # デフォルトは0.1 → 小さくして統計を安定させる

    for param in model.parameters():
        param.requires_grad = False
    logging.info("All layers initially frozen.")

    # 2. 'conv1'と'fc'層など、フェーズ1で学習させたい層をアンフリーズする
    # ResNetModelの実装に依存しますが、ここでは一般的なタスク固有層を想定します
    for name, param in model.named_parameters():
        # 'layer'を含まない層（conv1, bn1, fcなど）をアンフリーズ
        if 'layer' not in name:
            param.requires_grad = True
            logging.info(f"Unfreezing layer: {name}")

    # フェーズ1用の学習対象パラメータを抽出
    params_step1 = list(filter(lambda p: p.requires_grad, model.parameters()))
    logging.info(f"Phase 1 (Epoch 0 - {START_EPOCH_2-1}) training layers count: {len(params_step1)}")

    # 3. フェーズ2（START_EPOCH_2以降）で使う、全ての層のパラメータリストを準備
    # 全ての層をアンフリーズした後のパラメータを準備します。
    # このリストはフェーズ2でのみ使用します。
    logging.info(f"Phase 2 (Epoch {START_EPOCH_2} onwards) training layers count: {len(list(model.parameters()))}")

    optimizer_step1 = torch.optim.Adam(params_step1, lr=LEARNING_RATE, weight_decay=WEIGHT)
    optimizer_step2 = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE/2, weight_decay=WEIGHT)
    optimizer_step3 = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE/10, weight_decay=WEIGHT)

    scheduler_step2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_step2,
        T_max=START_EPOCH_3 - START_EPOCH_2,  # Step2 の期間
        eta_min=LEARNING_RATE/5               # Step3 LR に近い値まで下げる
    )

    scheduler_step3 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_step3,
        T_max=EPOCHS - START_EPOCH_3,
        eta_min=LEARNING_RATE/20              # 最終的にさらに小さく
    )

    # 損失関数はRGBベクトル間の角度誤差
    loss_fn = mixed_loss


    # 学習記録用リスト
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    all_start_time =time.time()

    # Epochループ
    for epoch in range(EPOCHS):
        logging.info(f"==== Epoch {epoch+1}/{EPOCHS} ====")
        epoch_start_time = time.time()
    

        if(epoch == START_EPOCH_2):
            # フェーズ2の開始エポック
            for param in model.parameters():  # 修正: model.parameters()をイテレートする
                param.requires_grad = True
            logging.info("--- PHASE 2 START: All layers unfrozen for fine-tuning. ---")
        
        # 学習フェーズに応じて optimizer と scheduler を使い分け
        if epoch < START_EPOCH_2:
            train_loss, train_batch_losses = train_one_epoch(model, train_loader, optimizer_step1, loss_fn)
            # Step1 は scheduler を使わない（固定LR）
        
        elif epoch < START_EPOCH_3:
            train_loss, train_batch_losses = train_one_epoch(model, train_loader, optimizer_step2, loss_fn)
            scheduler_step2.step()  # ← Step2 では epoch ごとに LR を更新
        
        else:
            train_loss, train_batch_losses = train_one_epoch(model, train_loader, optimizer_step3, loss_fn)
            scheduler_step3.step()  # ← Step3 では epoch ごとに LR を更新

        val_loss, val_batch_losses = evaluate(model, val_loader, loss_fn)
        val_angular_errors = np.array(val_batch_losses)
        epoch_end_time = time.time()
        logging.info(f"Total epoch time: {epoch_end_time - epoch_start_time:.2f} sec")
        logging.info(f"Loss: Train = {train_loss:.4f}, Val = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 検証損失が最小を更新した場合のみモデルを保存
            # ファイル名を 'best_resnet_model.pth' にして、最終保存と区別
            torch.save(model.state_dict(), OUTPUT_DIR / f'model.pth')
            logging.info(f"NEW BEST MODEL SAVED! Val Loss: {best_val_loss:.4f} !!!!!!")

            # 平均と中央値の計算を追加
            mean_error = np.mean(val_angular_errors) # 角度誤差のMean
            median_error = np.median(val_angular_errors)
            percentile_95 = np.percentile(val_angular_errors, 95)
            
            # ログにMeanも出力
            logging.info(f"Val Stats: Mean={mean_error:.4f}, Median={median_error:.4f}, 95-P={percentile_95:.4f}")
            
            # TensorBoard に Meanも記録
            writer.add_scalar('AngularErrorStats/Mean', mean_error, epoch+1)
            writer.add_scalar('AngularErrorStats/Median', median_error, epoch+1)
            writer.add_scalar('AngularErrorStats/95th_Percentile', percentile_95, epoch+1)

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