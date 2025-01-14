# continual_learning_dataloader.py

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from typing import Tuple, List, Dict
from collections import Counter

from utils.colored_prints import print_info
from utils.constants import DATA_ROOT

# Define the mapping from task_id to class labels
TASK_CLASSES: Dict[int, List[int]] = {
    # Task 0: Not part of the continual learning setup. This is just to test the model.
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # CL setup: Task 1, Task 2, Task 3, Task 4
    1: [0, 2, 4, 6],  # T-Shirt/Top, Pullover, Coat, Shirt
    2: [1, 3],  # Trouser, Dress
    3: [5, 7, 9],  # Sandals, Sneaker, Ankle boots
    4: [8],  # Bag
}


def get_dataloaders(
    task_id: int, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns the training and testing DataLoaders for a given task.

    Args:
        task_id (int): The ID of the task (e.g., 1, 2, 3, 4).
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and testing DataLoaders.
    """

    if task_id == 0:
        print_info(
            f"Using task_id=0. This uses all classes. This is only meant to test the model. This is NOT a continual learning scenario!"
        )

    if task_id not in TASK_CLASSES:
        raise ValueError(
            f"Invalid task_id {task_id}. Must be one of {list(TASK_CLASSES.keys())}."
        )

    classes = TASK_CLASSES[task_id]

    # Define the transformation (you can add more transforms if needed)
    transform = transforms.ToTensor()

    # Load the full training and testing datasets
    train_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, transform=transform, download=True
    )

    # Function to filter dataset indices based on desired classes
    def filter_indices(
        dataset: torchvision.datasets.FashionMNIST, classes: List[int]
    ) -> List[int]:
        """
        Filters the dataset and returns indices of samples belonging to the specified classes.

        Args:
            dataset (torchvision.datasets.FashionMNIST): The dataset to filter.
            classes (List[int]): The list of class labels to include.

        Returns:
            List[int]: A list of indices for samples belonging to the specified classes.
        """
        return [idx for idx, label in enumerate(dataset.targets) if label in classes]

    # Get the filtered indices for training and testing datasets
    train_indices = filter_indices(train_dataset, classes)
    test_indices = filter_indices(test_dataset, classes)

    # Create Subsets for the current task
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=2,  # Number of subprocesses for data loading
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader


def count_classes(dataset_subset: Subset) -> Dict[int, int]:
    """
    Counts the number of samples per class in a given dataset subset.

    Args:
        dataset_subset (Subset): The dataset subset to analyze.

    Returns:
        Dict[int, int]: A dictionary mapping class labels to their respective counts.
    """
    # Access the original dataset's targets and the subset's indices
    targets = dataset_subset.dataset.targets[dataset_subset.indices]
    # If targets are tensors, convert to list
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    # Count the occurrences of each class
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
        breakpoint()
        # Access the subsets directly to count classes without iterating through DataLoader
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
