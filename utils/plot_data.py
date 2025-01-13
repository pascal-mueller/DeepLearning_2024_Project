import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_dataloaders(
    dataloaders,
    results_dir,
    filename,
    title="Subset Data Visualization",
    xlabel="X",
    ylabel="Y",
    grid=True,
):
    save_path = os.path.join(results_dir, filename)

    single_plot_size = 5
    fig_width = single_plot_size * len(dataloaders)
    fig, axes = plt.subplots(
        1, len(dataloaders), figsize=(fig_width, single_plot_size), sharey=True
    )

    for task_id, (dataloader, ax) in enumerate(zip(dataloaders, axes)):
        indices = dataloader.dataset.indices

        # Get the data and labels for the subset
        data = dataloader.dataset.dataset.data[indices].view(-1, 2)
        labels = dataloader.dataset.dataset.labels[indices].argmax(dim=1).numpy()

        labels = np.repeat(labels, 4)

        colors = np.where(labels == 0, "blue", "red")

        ax.scatter(data[:, 0], data[:, 1], color=colors, alpha=0.5)
        ax.set_title(f"Task {task_id + 1} Data")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(grid)

    fig.suptitle(title)

    plt.savefig(save_path)
    plt.show()
