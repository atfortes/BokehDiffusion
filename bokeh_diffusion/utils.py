import cv2
import numpy as np
from PIL import Image


def color_transfer_lab(source_pil, target_pil, adjust_std=True):
    source_np = np.array(source_pil, dtype=np.uint8)
    target_np = np.array(target_pil, dtype=np.uint8)
    source_lab = cv2.cvtColor(source_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_src, std_src = cv2.meanStdDev(source_lab)
    mean_tgt, std_tgt = cv2.meanStdDev(target_lab)
    eps = 1e-6
    for c in range(3):
        t_chan = target_lab[..., c]
        t_chan -= mean_tgt[c][0]
        if adjust_std and std_tgt[c][0] > eps:
            t_chan *= (std_src[c][0] / std_tgt[c][0])
        t_chan += mean_src[c][0]
        target_lab[..., c] = t_chan
    target_lab = np.clip(target_lab, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(target_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(matched)
