import numpy as np
import torch

class HistogramDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, erase_prob=0.0, erase_size=0.0, rng=None):
        self.X = X
        self.y = y
        self.erase_prob = erase_prob
        self.erase_size = int(erase_size)
        self.rng = rng or np.random.default_rng()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.rng.random() < self.erase_prob:
            x = self.random_erase(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def random_erase(self, img):
        x = self.rng.integers(0, 224 - self.erase_size)
        y = self.rng.integers(0, 224 - self.erase_size)
        img[y:y+self.erase_size, x:x+self.erase_size] = 0
        return img


