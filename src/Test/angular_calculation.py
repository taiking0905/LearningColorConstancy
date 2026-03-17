import numpy as np

e = np.array([0.232, 0.457, 0.311])
g = np.array([0.224, 0.505, 0.271])

dot = np.dot(e, g)
norms = np.linalg.norm(e) * np.linalg.norm(g)
angle_rad = np.arccos(np.clip(dot / norms, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)

print("角度誤差:", angle_deg)
