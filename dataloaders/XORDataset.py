import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from collections import Counter

def get_dataloaders(
    train_batch_size=32,
    test_batch_size=32,
    shuffle_train=True,
    split_ratio=0.8,
    n_samples=2000
):
    train_loaders = {}
    test_loaders = {}
    for task_id in range(1, 4):
        dataset = XORDataset(task_id, n_samples=n_samples)
        labels = dataset.labels.argmax(dim=1).numpy()

        class0_indices = np.where(labels == 0)[0]
        class1_indices = np.where(labels == 1)[0]

        np.random.shuffle(class0_indices)
        np.random.shuffle(class1_indices)

        split_class0 = int(len(class0_indices) * split_ratio)
        split_class1 = int(len(class1_indices) * split_ratio)

        train_indices = np.concatenate([class0_indices[:split_class0], class1_indices[:split_class1]])
        test_indices = np.concatenate([class0_indices[split_class0:], class1_indices[split_class1:]])

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=shuffle_train)
        test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

        train_loaders[task_id] = train_loader
        test_loaders[task_id] = test_loader

    return train_loaders, test_loaders


class XORDataset(Dataset):
    def __init__(self, task_id, n_samples=50, noise=0.1):
        self.data = torch.zeros(n_samples, 4, 2)
        self.task_id = task_id
        self.labels = torch.zeros(n_samples)

        if task_id == 1:  # XOR on diagonals
            for i in range(n_samples):
                if i < n_samples / 2:  # Label 1
                    points = torch.tensor(
                        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]]
                    )
                else:  # Label 0
                    points = torch.tensor(
                        [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.75, 0.25]]
                    )
                self.data[i] = points + torch.randn_like(points) * noise
                self.labels[i] = 1 if i < n_samples / 2 else 0

        elif task_id == 2:  # Square vs Diamond
            for i in range(n_samples):
                if i < n_samples / 2:  # Label 1: Diamond
                    points = torch.tensor(
                        [[0.0, 0.5], [0.5, 0.0], [0.5, 1.0], [1.0, 0.5]]
                    )
                else:  # Label 0: Square
                    points = torch.tensor(
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                    )
                self.data[i] = points + torch.randn_like(points) * noise
                self.labels[i] = 1 if i < n_samples / 2 else 0

        elif task_id == 3:  # Close pairs vs Distant pairs
            for i in range(n_samples):
                if i < n_samples / 2:  # Label 1: Close pairs
                    points = torch.tensor(
                        [[0.3, 0.3], [0.4, 0.4], [0.6, 0.6], [0.7, 0.7]]
                    )
                else:  # Label 0: Distant pairs
                    points = torch.tensor(
                        [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]]
                    )
                self.data[i] = points + torch.randn_like(points) * noise
                self.labels[i] = 1 if i < n_samples / 2 else 0

        def to_one_hot(x):
            n = len(x)
            result = torch.zeros((n, 2))
            result[range(n), x] = 1
            return result

        self.labels = to_one_hot(self.labels.int())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.task_id

    def plot(self):
        # Separate data points by label
        task_data = self.data.view(-1, 4, 2).numpy()
        labels = self.labels.argmax(dim=1).numpy()

        for i, (points, label) in enumerate(zip(task_data, labels)):
            points = points.reshape(-1, 2)
            color = "blue" if label == 0 else "red"
            plt.scatter(points[:, 0], points[:, 1], color=color, alpha=0.5)

        plt.title(f"Task {self.task_id} Data Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


# Run this file directly to test for balancedness
if __name__ == "__main__":
    batch_size = 32
    split_ratio = 0.8  # 80% training, 20% testing
    n_samples = 2000

    train_dataloaders, test_dataloaders = get_dataloaders(
        test_batch_size=batch_size,
        train_batch_size=batch_size,
        split_ratio=split_ratio,
        n_samples=n_samples
    )

    for task_id in range(1, 4):
        train_loader, test_loader = train_dataloaders[task_id], test_dataloaders[task_id]

        # Function to count classes in a DataLoader
        def count_classes(dataloader):
            class_counts = Counter()
            total_samples = 0
            for _, batch_labels, _ in dataloader:
                labels = batch_labels.argmax(dim=1).numpy()
                class_counts.update(labels)
                total_samples += len(labels)
            return class_counts, total_samples

        # Count classes in training and testing sets
        train_class_counts, train_total = count_classes(train_loader)
        test_class_counts, test_total = count_classes(test_loader)

        # Calculate percentages
        train_class0_pct = (
            (train_class_counts[0] / train_total) * 100 if train_total > 0 else 0
        )
        train_class1_pct = (
            (train_class_counts[1] / train_total) * 100 if train_total > 0 else 0
        )

        test_class0_pct = (
            (test_class_counts[0] / test_total) * 100 if test_total > 0 else 0
        )
        test_class1_pct = (
            (test_class_counts[1] / test_total) * 100 if test_total > 0 else 0
        )

        # Print the percentages
        print(f"Task {task_id} Class Distribution:")
        print("Training Set:")
        print(f"  Class 0: {train_class_counts[0]} samples ({train_class0_pct:.2f}%)")
        print(f"  Class 1: {train_class_counts[1]} samples ({train_class1_pct:.2f}%)")
        print("Testing Set:")
        print(f"  Class 0: {test_class_counts[0]} samples ({test_class0_pct:.2f}%)")
        print(f"  Class 1: {test_class_counts[1]} samples ({test_class1_pct:.2f}%)\n")
