import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_three_lines(data: np.ndarray, title: str, save_path: str):
    x = np.arange(0.05, 2.05, 0.05)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mapping = {0: 'R', 1: 'G', 2: 'B'}

    for idx in range(3):
        axes[idx].plot(x, data[:, idx], linewidth=2)
        axes[idx].set_title(f'{title} {mapping[idx]} channel')
        axes[idx].set_xlabel('sigma value')
        axes[idx].set_ylabel('Illuminant feature')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_combined_line(data: np.ndarray, title: str, save_path: str):
    """
    data: shape (40,3) 的 mean_curve
    """
    # 將 R、G、B 三通道等權重平均
    combined = data.mean(axis=1)  # shape (40,)
    x = np.arange(0.05, 2.05, 0.05)

    plt.figure(figsize=(6, 4))
    plt.plot(x, combined, linewidth=2)
    plt.title(title)
    plt.xlabel('sigma value')
    plt.ylabel('Illuminant feature (R+G+B)/3')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 1. Load your JSON of index‐lists
with open('arr_reference.json', 'r') as f:
    refs = json.load(f)

feature_dir = "./illuminant_features"

# 2. For each of the four groups, compute mean curve and plot
for group in ['all', 'best25', 'worst25', 'others']:
    ids = refs[group]
    mats = []
    for img_id in ids:
        fn = os.path.join(feature_dir, f"{img_id}.npy")
        if not os.path.exists(fn):
            print(f"Warning: {fn} not found, skipping.")
            continue
        mats.append(np.load(fn))  # each is shape (40,3)

    if not mats:
        raise RuntimeError(f"No .npy files loaded for group {group}!")

    arr = np.stack(mats, axis=0)      # (N_images, 40, 3)
    mean_curve = arr.mean(axis=0)     # (40, 3)

    # 繪製三條線
    plot_three_lines(
        mean_curve,
        title=f"{group} (mean over {len(mats)} samples)",
        save_path=f"fig_{group}.png"
    )
    # 繪製加權單線
    plot_combined_line(
        mean_curve,
        title=f"{group} combined (mean over {len(mats)} samples)",
        save_path=f"fig_{group}_combined.png"
    )
