import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import sqlite3

from dataloaders.FashionMNISTDataset import get_dataloaders
from nn.FashionMNIST import FashionMNIST
from utils.random_conf import ensure_deterministic

device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def train_model(
    model, train_loader, epochs=10, lr=0.001, l1_lambda=0.0, verbose_leve=-1
):
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        # (Pdb) data.shape
        # torch.Size([64, 1, 28, 28])
        # (Pdb) labels.shape
        # torch.Size([64])
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            if l1_lambda > 0.0:
                l1_penalty = torch.tensor(0.0)
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                loss += l1_penalty * l1_lambda

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)

        if verbose_leve >= 1:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f}")


def test_model(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)  # shape: [batch_size, num_classes]
            _, predicted = torch.max(outputs, dim=1)  # get class indices

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
    device=torch.device("cpu"),
):
    if plot_data or plot_losses or plot_fim:
        raise NotImplementedError("Plots are not implemented for this experiment yet.")

    num_epochs = params.get("num_epochs", 10)
    learning_rate = params.get("learning_rate", 0.001)
    l1_lambda = params.get("l1_lambda", 0.0)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model = FashionMNIST().to(device)

    accuracies = {"Task1": [], "Task2": [], "Task3": [], "Task4": []}

    loaders = {taskid: get_dataloaders(taskid) for taskid in range(1, 5)}

    for taskid in range(1, 5):
        if verbose_level >= 0:
            print(f"=== Training on Task {taskid} ===")
        train_loader, _ = loaders[taskid]

        train_model(
            model,
            train_loader,
            epochs=num_epochs,
            lr=learning_rate,
            l1_lambda=l1_lambda,
            verbose_leve=verbose_level,
        )

        # Evaluate on all previous tasks
        for i in range(1, taskid + 1):
            _, test_loader = loaders[i]
            acc = test_model(model, test_loader)
            if verbose_level >= 0:
                print(
                    f"After training Task {taskid}, accuracy on Task {i}: {100 * acc:.3f}%"
                )
            accuracies[f"Task{i}"].append(acc)

    if verbose_level >= 0:
        print("\n=== Final Accuracies ===")
        print("Task 1 Accuracies:", accuracies["Task1"])
        print("Task 2 Accuracies:", accuracies["Task2"])
        print("Task 3 Accuracies:", accuracies["Task3"])
        print("Task 4 Accuracies:", accuracies["Task4"])

    avg_perf = (
        sum(
            accuracies["Task1"]
            + accuracies["Task2"]
            + accuracies["Task3"]
            + accuracies["Task4"]
        )
        / 10
    )

    return params, accuracies, avg_perf


# Objective function for Optuna
def objective(trial, run_name):
    # Define the hyperparameter search space
    num_epochs = trial.suggest_int("num_epochs", 10, 200)
    inner_epochs = trial.suggest_int("inner_epochs", 10, 200)
    learning_rate = trial.suggest_float("learning_rate", 1e-9, 1e-5, log=True)
    control_lr = trial.suggest_float("control_lr", 1e-8, 1e-3, log=True)
    control_threshold = trial.suggest_float("control_threshold", 1e-14, 1e-6, log=True)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-6, 2e-1, log=True)

    # Run the model with the sampled parameters
    params = {
        "num_epochs": num_epochs,
        "inner_epochs": inner_epochs,
        "learning_rate": learning_rate,
        "control_lr": control_lr,
        "control_threshold": control_threshold,
        "l1_lambda": l1_lambda,
    }
    _, task_performance, avg_perf = run_experiment(
        params, run_name=run_name, verbose_level=-1
    )

    # Evaluation metric: Average accuracy across tasks
    # avg_accuracy = np.mean(
    #     [
    #         acc
    #         for task_results in task_performance.values()
    #         for acc in task_results.values()
    #     ]
    # )

    # Goal is to maximize avg_accuracy
    return avg_perf


# Run the Optuna study
def run_optuna_study(
    run_name,
    num_trials,
    num_cpus,
    dbname="paramsearch",
    study_name="hyperparameter_optimization",
):
    from functools import partial

    assert num_cpus <= 48, "Max 48 CPUs supported for SQLite storage"

    results_dir = os.path.join("results", run_name)

    results_dir = os.path.join("results/bp_full_XOR", run_name)
    os.makedirs(results_dir, exist_ok=True)
    db_path = os.path.join(results_dir, dbname)

    # Use SQLite as shared storage for parallel workers

    storage = (
        f"sqlite:///{db_path}?check_same_thread=False&pool_size=20&max_overflow=48"
    )
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    objective_ = partial(objective, run_name=run_name)

    # Run optimization with parallel trials
    study.optimize(objective_, n_trials=num_trials, n_jobs=num_cpus)

    # Print and return the best results
    print("Best parameters:", study.best_params)
    print("Best value (accuracy):", study.best_value)
    print("Best trial:", study.best_trial)

    return study


if __name__ == "__main__":
    from nn.Net import Net

    device = torch.device("cpu")

    ensure_deterministic()
    params_75_to_80 = {
        "num_epochs": 10,
        "learning_rate": 0.001,
        "l1_lambda": 1e-5,
    }
    run_experiment(params_75_to_80, run_name="FashionMNIST", verbose_level=0)
    quit()

    # Taks 0 includes all classes.
    print(f"Running FashionMNIST experiment on full data. This is not a CL scenario.")

    train_loader, test_laoder = get_dataloaders(0)
    model = Net(input_size=784, hidden_size=100, output_size=10, softmax=False)

    train_model(model, train_loader, epochs=10, lr=0.001, l1_lambda=0.0)
    acc = test_model(model, test_laoder)
    print(f"Accuracy: {acc:.4f}")
