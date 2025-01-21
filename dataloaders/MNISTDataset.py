import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TASK_CLASSES = {
    0: list(range(10)),  # all classes 0..9 (could be used for a test scenario)
    1: [0, 2, 4, 6],
    2: [1, 3],
    3: [5, 7, 9],
    4: [8],
}


def get_dataloaders(task_id, train_batch_size=32, test_batch_size=32):
    """
    Returns train_loader and test_loader for the union of classes from tasks 1..task_id.
    If task_id=0, we default to 'all classes' (TASK_CLASSES[0]).
    """
    # Basic transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load full MNIST datasets
    train_dataset_full = datasets.MNIST(
        root="local_data", train=True, transform=transform, download=True
    )
    test_dataset_full = datasets.MNIST(
        root="local_data", train=False, transform=transform, download=True
    )

    # Determine which classes to use:
    if task_id == 0:
        # For example, use all classes [0..9]
        union_classes = TASK_CLASSES[0]
    else:
        # Union of tasks [1..task_id]
        union_classes = []
        for tid in range(1, task_id + 1):
            union_classes.extend(TASK_CLASSES[tid])
        union_classes = list(set(union_classes))  # deduplicate

    # Filter the datasets by 'union_classes'
    train_indices = [
        i for i, y in enumerate(train_dataset_full.targets) if y in union_classes
    ]
    test_indices = [
        i for i, y in enumerate(test_dataset_full.targets) if y in union_classes
    ]

    # Create subsets
    train_subset = Subset(train_dataset_full, train_indices)
    test_subset = Subset(test_dataset_full, test_indices)

    # Wrap in DataLoader
    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader
