import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TASK_CLASSES = {
    # All data
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # CL tasks
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
    4: [6, 7],
    5: [8, 9],
}


def get_dataloaders(
    train_batch_size=32,
    test_batch_size=32,
    shuffle_train=True,
    return_indices=False,
    return_masks=False,
    seperate_data=False,
):
    """
    Returns train and test dataloaders for all tasks as well as god mode.
    Optionally also returns the masks and indices for test and train data for
    each task.

    idx: 0 = all data
    idx: 1-4 = task 1-4
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

    # Get labels
    train_labels = train_dataset_full.targets
    test_labels = test_dataset_full.targets

    train_masks = []
    test_masks = []
    train_indices = []
    test_indices = []
    train_dataloaders = []
    test_dataloaders = []

    # Create masks, indices and dataloaders for each task
    for i, task_classes in enumerate(TASK_CLASSES.values()):
        # Create masks
        train_mask = torch.isin(train_labels, torch.tensor(task_classes))
        test_mask = torch.isin(test_labels, torch.tensor(task_classes))

        # Create indices
        train_idxs = torch.nonzero(train_mask).squeeze()
        test_idxs = torch.nonzero(test_mask).squeeze()

        train_masks.append(train_masks)
        test_masks.append(test_masks)
        train_indices.append(train_idxs)
        test_indices.append(test_idxs)

        if seperate_data or i == 0:
            # Create subsets
            train_subset = Subset(train_dataset_full, train_idxs)
            test_subset = Subset(test_dataset_full, test_idxs)

            # Create dataloaders
            train_dataloader = DataLoader(
                train_subset, batch_size=train_batch_size, shuffle=shuffle_train
            )
            test_dataloader = DataLoader(
                test_subset, batch_size=test_batch_size, shuffle=False
            )
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

    if not seperate_data:
        train_idxs = []
        test_idxs = []
        # next() skips the first element
        for train_indices, test_indices in zip(train_indices[1:], test_indices[1:]):
            train_idxs += train_indices.tolist()
            test_idxs += test_indices.tolist()

            # Create subsets
            train_subset = Subset(train_dataset_full, torch.tensor(train_idxs))
            test_subset = Subset(test_dataset_full, torch.tensor(test_idxs))

            # Create dataloaders
            train_dataloader = DataLoader(
                train_subset, batch_size=train_batch_size, shuffle=shuffle_train
            )
            test_dataloader = DataLoader(
                test_subset, batch_size=test_batch_size, shuffle=False
            )
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

    if return_indices and return_masks:
        return (
            train_dataloaders,
            test_dataloaders,
            train_indices,
            test_indices,
            train_masks,
            test_masks,
        )

    if return_indices:
        return train_dataloaders, test_dataloaders, train_indices, test_indices

    if return_masks:
        return train_dataloaders, test_dataloaders, train_masks, test_masks

    return train_dataloaders, test_dataloaders
