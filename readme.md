# Learning Color Constancy  

本プロジェクトは、機械学習をして色恒常性を手に入れる研究です。  

簡単に色恒常性について学べます。また修正後の画像を見比べることができます。  
🔗 https://taiking0905.github.io/LearningColorConstancy_exhibit/  

---

## ✅ 環境構築手順

### Python仮想環境の作成

```bash
python -m venv LearningColorConstancy_env   

.\LearningColorConstancy_env\Scripts\Activate   

pip install -r requirements.txt
"E:\LearningColorConstancy\histogram"

tensorboard --logdir=\outputs\tb_logs\
```

## 🔧 インストール手順（GPU対応）

このプロジェクトでは PyTorch の CUDA 対応版（12.8）を使用しています。  
以下のコマンドでインストールしてください：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

進捗報告:https://field-motorcycle-315.notion.site/2492c8c0fa7c8083a14ff4999e612917

---
