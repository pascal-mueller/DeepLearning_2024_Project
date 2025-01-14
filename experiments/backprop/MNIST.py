import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders.MNISTDataset import get_dataloaders
from nn.MNISTModel import MNISTModel

# BEST_PARAMS = {
#     "num_epochs": 600,
#     "learning_rate": 0.1,
#     "l1_lambda": 1e-5,
# }


def train_model(model, train_loader, epochs=10, lr=0.001, l1_lambda=0.0, device="cpu"):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, y, _) in enumerate(train_loader):
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            if l1_lambda > 0.0:
                l1_penalty = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                loss += l1_penalty * l1_lambda

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        # print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f}")


def test_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)  # shape: [batch_size, num_classes]
            _, predicted = torch.max(outputs, dim=1)  # get class indices
            labels = y

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def run_experiment(
    params,
    run_name,
    seed=0,
    plot_data=False,
    plot_losses=False,
    plot_fim=False,
    verbose_level=-1,
):
    if plot_data or plot_losses or plot_fim:
        raise NotImplementedError("Plots are not implemented for this experiment yet.")

    num_epochs = params.get("num_epochs", 10)
    learning_rate = params.get("learning_rate", 0.01)
    l1_lambda = params.get("l1_lambda", 0.0)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model = MNISTModel()

    accuracies = {"Task1": [], "Task2": [], "Task3": [], "Task4": []}

    loaders = {taskid: get_dataloaders(taskid) for taskid in range(1, 5)}

    for taskid in range(1, 5):
        if verbose_level > 0: print(f"=== Training on Task {taskid} ===")
        train_loader, _ = loaders[taskid]

        train_model(
            model,
            train_loader,
            epochs=num_epochs,
            lr=learning_rate,
            l1_lambda=l1_lambda,
        )

        # Evaluate on all previous tasks
        for i in range(1, taskid + 1):
            _, test_loader = loaders[i]
            acc = test_model(model, test_loader)
            if verbose_level > 0: print(f"After training Task {taskid}, accuracy on Task {i}: {100 * acc:.3f}%")
            accuracies[f"Task{i}"].append(acc)

    if verbose_level > 0:
        print("\n=== Final Accuracies ===")
        for i in range(1, 5):
            print(f"Task {i} Accuracies:", accuracies[f"Task{i}"])

    avg_perf = sum(sum(accuracies[f"Task{i}"]) for i in range (1, 5)) / 4

    return params, accuracies, avg_perf


if __name__ == "__main__":
    # params_75_to_80 should give between 75% and 80%
    params_75_to_80 = {
        "num_epochs": 10,
        "learning_rate": 0.001,
        "l1_lambda": 1e-5,
    }

    # avgs = [run_experiment(params_75_to_80, run_name=f"Run {i+1}")[2] for i in range(10)]
    _, _, avg = run_experiment(params_75_to_80, run_name="test_run", verbose_level=1)
    print(avg)
    # print(avgs, avg)
    # print(sum(avgs)/len(avgs))
