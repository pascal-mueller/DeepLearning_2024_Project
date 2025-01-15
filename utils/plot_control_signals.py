import os
import torch
import matplotlib.pyplot as plt


def plot_control_signals(control_signals, results_dir, filename):
    save_path = os.path.join(results_dir, filename)

    control_signals = torch.tensor(control_signals)

    hidden_mean = control_signals[:, 0]
    output_mean = control_signals[:, 1]

    plt.plot(range(len(hidden_mean)), hidden_mean, "r")
    plt.plot(range(len(output_mean)), output_mean, "b")

    plt.savefig(save_path)
    plt.show()
