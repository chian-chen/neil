import matplotlib.pyplot as plt
import numpy as np



import numpy as np
import matplotlib.pyplot as plt

def plot_three_lines(data: np.ndarray):

    K = data.shape[0]
    x = np.arange(0.05, 2.05, 0.05)

    # Create a single figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mapping = {0: 'R', 1: 'G', 2: 'B'}

    for idx in range(3):
        axes[idx].plot(x, data[:, idx])
        axes[idx].set_title(f'{mapping[idx]} Channel')
        axes[idx].set_xlabel('sigma value')
        axes[idx].set_ylabel('illuminant feature')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig('three_lines_plot.png')




path = "./illuminant_features/1.npy"
illuminants = np.load(path)
print(illuminants.shape)
# plot_three_lines(illuminants)




