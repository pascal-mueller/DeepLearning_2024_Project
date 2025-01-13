import os
import matplotlib.pyplot as plt


def plot_losses(all_losses, save_path):
    save_path = os.path.join(save_path, "tl_full_XOR_losses.png")

    plt.plot(all_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over all tasks")
    plt.savefig(save_path)
    plt.show()
