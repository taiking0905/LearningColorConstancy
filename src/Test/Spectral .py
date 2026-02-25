import numpy as np
import matplotlib.pyplot as plt

wl = np.linspace(400, 700, 300)

def gaussian(mu, sigma):
    return np.exp(-0.5 * ((wl - mu)/sigma)**2)

S_R1 = gaussian(630, 30)
S_R2 = gaussian(640, 30)

# ---- ここがポイント ----
threshold = 0.6  # この値以下は見せない
S_R1_plot = np.where(S_R1 >= threshold, S_R1, np.nan)
S_R2_plot = np.where(S_R2 >= threshold, S_R2, np.nan)

plt.figure(figsize=(6,4))
plt.plot(wl, S_R1_plot, 'r', linewidth=3)
plt.plot(wl, S_R2_plot, 'r--', linewidth=3)

plt.xlim(560, 670)      # R付近だけ拡大
plt.ylim(0.6, 1.05)     # 頂点周辺のみ
plt.xlabel("Wavelength (nm)")
plt.ylabel("Sensitivity")
plt.title("Difference in Red Channel Peak Only")
plt.legend()
plt.tight_layout()
plt.show()
