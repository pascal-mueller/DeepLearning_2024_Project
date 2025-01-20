import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
from typing import Tuple, List, Dict, Union
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from utils.constants import DATA_ROOT

# Define the mapping from task_id to class labels
TASK_CLASSES: Dict[int, List[int]] = {
    # Task 0: Not part of the continual learning setup. This is just to test the model.
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    0: list(range(10)),
    1: [0, 2, 4, 6],
    2: [1, 3],
    3: [5, 7, 9],
    4: [8],
}


class MNISTDataset(Dataset):
    def __init__(self, task_id, train=True, transform=None):
        self.transform = transform
        self.task_id = task_id

        self.mnist_data = datasets.MNIST(
            root=DATA_ROOT,
            train=train,
            download=True,
        )

        data, targets = self.mnist_data.data, self.mnist_data.targets

        digits = TASKS[task_id]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for digit in digits:
            mask |= targets == digit

        self.data = data[mask]
        self.labels = targets[mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index].float()
        y = int(self.labels[index])

        x = x.unsqueeze(0)

        # If transform is defined (like normalization, etc.), apply it
        if self.transform:
            x = self.transform(x)

        return x, y


def get_dataloaders(task_id, batch_size=32):
    # Typical MNIST transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load the full training and testing datasets
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # Function to filter dataset indices based on desired classes
    def filter_indices(
        dataset: torchvision.datasets.MNIST, classes: List[int]
    ) -> List[int]:
        return [idx for idx, label in enumerate(dataset.targets) if label in classes]

    # Filter indices for the current task
    train_indices = filter_indices(train_dataset, classes)
    test_indices = filter_indices(test_dataset, classes)

    # Create filtered data and move to GPU
    train_data = train_dataset.data[train_indices].to(device).float()

    train_targets = train_dataset.targets[train_indices].to(device)

    test_data = test_dataset.data[test_indices].to(device).float()
    test_targets = test_dataset.targets[test_indices].to(device)

    # Create TensorDatasets using GPU-backed data
    train_subset = TensorDataset(train_data, train_targets)
    test_subset = TensorDataset(test_data, test_targets)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=train_batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=0,  # No extra processes needed since data is on GPU
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=test_batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=0,
    )

    return train_loader, test_loader


def count_classes(dataset: TensorDataset) -> Dict[int, int]:
    """
    Counts the number of samples per class in a given dataset.

    Args:
        dataset (TensorDataset): The dataset to analyze.

    Returns:
        Dict[int, int]: A dictionary mapping class labels to their respective counts.
    """
    targets = dataset.tensors[1].tolist()  # Extract labels
    class_counts = Counter(targets)
    return dict(class_counts)


def is_balanced(class_counts: Dict[int, int], tolerance: float = 0.05) -> bool:
    """
    Checks if the class counts are balanced within a specified tolerance.

    Args:
        class_counts (Dict[int, int]): A dictionary mapping class labels to their counts.
        tolerance (float): The allowed relative difference between class counts.

    Returns:
        bool: True if balanced, False otherwise.
    """
    counts = list(class_counts.values())
    if not counts:
        return False
    mean_count = sum(counts) / len(counts)
    for count in counts:
        if abs(count - mean_count) / mean_count > tolerance:
            return False
    return True


if __name__ == "__main__":
    """
    Tests the balancedness of each task's training and testing datasets.
    """
    # Define task IDs
    task_ids = list(TASK_CLASSES.keys())
    # Define batch size (arbitrary, since we're accessing dataset directly)
    batch_size = 64

    for task_id in task_ids:
        print(f"\n--- Task {task_id} ---")
        train_loader, test_loader = get_dataloaders(task_id, batch_size)

        # Access the TensorDatasets directly to count classes
        train_subset = train_loader.dataset
        test_subset = test_loader.dataset

        train_counts = count_classes(train_subset)
        test_counts = count_classes(test_subset)

        # Display class counts
        print("Training set class counts:")
        for cls in TASK_CLASSES[task_id]:
            print(f"  Class {cls}: {train_counts.get(cls, 0)} samples")
        print("Testing set class counts:")
        for cls in TASK_CLASSES[task_id]:
            print(f"  Class {cls}: {test_counts.get(cls, 0)} samples")

        # Check balancedness
        train_balanced = is_balanced(train_counts)
        test_balanced = is_balanced(test_counts)

        print(f"Training set balanced: {'Yes' if train_balanced else 'No'}")
        print(f"Testing set balanced: {'Yes' if test_balanced else 'No'}")

        # Optional: Assert balancedness
        assert train_balanced, f"Training set for Task {task_id} is not balanced."
        assert test_balanced, f"Testing set for Task {task_id} is not balanced."

    print("\nAll tasks have balanced training and testing datasets.")
