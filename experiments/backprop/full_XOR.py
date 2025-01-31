import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import sqlite3

from dataloaders.XORDataset import get_dataloaders
from nn.Net import Net
from utils.constants import RESULTS_ROOT
from utils.random_conf import ensure_deterministic

BEST_PARAMS = {
    "num_epochs": 600,
    "learning_rate": 0.1,
    "l1_lambda": 1e-5,
}


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
            _, labels = torch.max(y, dim=1)  # get class indices (one-hot)

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

    model = Net()

    accuracies = {"Task1": [], "Task2": [], "Task3": []}

    train_loaders, test_loaders = get_dataloaders()

    for taskid in range(1, 4):
        if verbose_level > 0:
            print(f"=== Training on Task {taskid} ===")
        train_loader = train_loaders[taskid]

        train_model(
            model,
            train_loader,
            epochs=num_epochs,
            lr=learning_rate,
            l1_lambda=l1_lambda,
        )

        # Evaluate on all previous tasks
        for i in range(1, taskid + 1):
            test_loader = test_loaders[i]
            acc = test_model(model, test_loader)
            if verbose_level > 0:
                print(
                    f"After training Task {taskid}, accuracy on Task {i}: {100 * acc:.3f}%"
                )
            accuracies[f"Task{i}"].append(acc)

    if verbose_level > 0:
        print("\n=== Final Accuracies ===")
        print("Task 1 Accuracies:", accuracies["Task1"])
        print("Task 2 Accuracies:", accuracies["Task2"])
        print("Task 3 Accuracies:", accuracies["Task3"])

    avg_perf = sum(accuracies["Task1"] + accuracies["Task2"] + accuracies["Task3"]) / 6

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

    results_dir = RESULTS_ROOT / "bp_full_XOR" / run_name
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
    ensure_deterministic()
    # params_75_to_80 should give between 75% and 80%
    params_75_to_80 = {
        "num_epochs": 200,
        "learning_rate": 0.01,
        "l1_lambda": 1e-5,
    }

    run_experiment(params_75_to_80, run_name="test_run", verbose_level=1)
