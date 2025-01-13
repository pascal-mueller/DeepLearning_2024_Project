import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_subset(
    subset,
    results_dir,
    filename,
    title="Subset Data Visualization",
    xlabel="X",
    ylabel="Y",
    grid=True,
):
    """
    Plots all the samples in a given Subset.
    """

    save_path = os.path.join(results_dir, filename)

    # subset.dataset is the original dataset
    # subset.indices is the list of indices that belong to this subset
    data_list = []
    label_list = []
    for idx in subset.indices:
        data, label, _ = subset.dataset[idx]
        data_list.append(data)  # shape [4, 2]
        label_list.append(label.argmax().item())

    data_array = torch.stack(data_list).numpy()  # shape [N, 4, 2]
    labels = np.array(label_list)

    # Plot each item in the subset
    for points, lbl in zip(data_array, labels):
        color = "blue" if lbl == 0 else "red"
        plt.scatter(points[:, 0], points[:, 1], color=color, alpha=0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.savefig(save_path)
    plt.show()
