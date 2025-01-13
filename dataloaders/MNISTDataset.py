from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

TASKS = {
    1: [0, 2, 4, 6],
    2: [1, 3],
    3: [5, 7, 9],
    4: [8]
}

class MNISTDataset(Dataset):
    def __init__(self, task_id, train=True, transform=None):
        self.transform = transform
        self.task_id = task_id

        self.mnist_data = datasets.MNIST(
            root="../local_data",  # or any other path
            train=train,
            download=True
        )

        data, targets = self.mnist_data.data, self.mnist_data.targets

        digits = TASKS[task_id]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for digit in digits:
            mask |= (targets == digit)

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

        return x, y, self.task_id


def get_dataloaders(task_id, batch_size=32):
    # Typical MNIST transformations
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
    ])

    train_dataset = MNISTDataset(task_id, train=True, transform=transform)
    test_dataset = MNISTDataset(task_id, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Run this file directly to test for balancedness
if __name__ == "__main__":
    for task_id in range(1, 5):
        batch_size = 32

        train_loader, test_loader = get_dataloaders(task_id, batch_size=batch_size)

        # Function to count classes in a DataLoader
        def count_classes(dataloader):
            class_counts = Counter()
            total_samples = 0
            for _, batch_labels, _ in dataloader:
                labels = batch_labels.numpy()
                class_counts.update(labels)
                total_samples += len(labels)
            return class_counts, total_samples

        # Count classes in training and testing sets
        train_class_counts, train_total = count_classes(train_loader)
        test_class_counts, test_total = count_classes(test_loader)

        print(f"Task {task_id} Class Distribution:")

        # Calculate percentages
        for key in train_class_counts.keys():
            train_rel = train_class_counts[key] / train_total * 100 if train_total > 0 else 0
            print(f"Train class {key}: {train_class_counts[key]} samples ({train_rel:.2f}%)")
            test_rel = test_class_counts[key] / test_total * 100 if test_total > 0 else 0
            print(f"Test class {key}: {test_class_counts[key]} samples ({test_rel:.2f}%)\n")