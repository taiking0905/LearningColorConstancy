# インストール

## テスト環境

- OS: Windows 11 Home
- GPU: NVIDIA RTX 2060
- CUDA: 12.8
- Python: 3.13.2
- PyTorch: 2.8.0 (cu128)
- torchvision: 0.22.1
- rawpy: 0.25.0
- OpenCV: 4.11.0.86
- scikit-learn: 1.6.1

## セットアップ

### 1. 仮想環境の作成

``` bash
python -m venv LearningColorConstancy_env
.\LearningColorConstancy_env\Scripts\activate
```

### 2. 依存ライブラリのインストール

``` bash
pip install -r requirements.txt
```

### 3. CUDA対応 PyTorch のインストール

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. 動作確認
CUDAが正しく認識されているか確認

``` bash
python -c "import torch; print(torch.cuda.is_available())"
```