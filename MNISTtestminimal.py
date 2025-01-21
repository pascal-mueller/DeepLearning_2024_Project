import torch
import torch.nn as nn
import os
import optuna
import sqlite3
from tqdm import tqdm
from nn.Net import Net
from nn.ControlNet import ControlNet
from utils.colored_prints import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataloaders.MNISTDataset import get_dataloaders, TASK_CLASSES

from utils.constants import DATA_ROOT


def print_green(text):
    green_color_code = "\033[92m"
    reset_color_code = "\033[0m"
    print(f"{green_color_code}{text}{reset_color_code}")


def evaluate_model(net, control_net, test_loader, useSignals=False):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            net.reset_control_signals()
            h1 = net.layer1(net.flatten(batch_data))
            output = net(batch_data)
            current_activities = torch.cat([net.flatten(batch_data), h1, output], dim=1)

            control_signals = control_net(current_activities)
            if useSignals:
                net.set_control_signals(control_signals)
            output = net(batch_data)

            predictions = output.max(dim=1).indices
            total += batch_labels.size(0)
            correct += (predictions == batch_labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def train_model(
    task_id,
    net,
    control_net,
    train_loader_god,
    criterion,
    control_optimizer,
    net_optimizer,
    control_threshold,
    num_epochs,
    inner_epochs,
    l1_lambda,
):
    task_classes = torch.tensor(TASK_CLASSES[task_id])

    pbar = tqdm(range(num_epochs), desc=f"Epochs", leave=False, disable=True)

    for epoch in pbar:
        # print(f"Press any key to start epoch {epoch}")
        # input()
        batch_losses = []

        for batch_data_god, batch_labels_god in train_loader_god:
            no_god_mask = torch.isin(
                batch_labels_god,
                torch.tensor(task_classes),
            )
            batch_data_nogod = batch_data_god[no_god_mask]
            batch_labels_nogod = batch_labels_god[no_god_mask]

            # Get current network activities
            with torch.no_grad():
                net.reset_control_signals()
                h1 = net.layer1(net.flatten(batch_data_god))
                output = net(batch_data_god)
                current_activities = torch.cat(
                    [net.flatten(batch_data_god), h1, output], dim=1
                )

            old_loss = float("inf")
            for inner_epoch in range(inner_epochs):
                control_optimizer.zero_grad()
                # net_optimizer.zero_grad()  # TODO: Do I need this?

                control_signals = control_net(current_activities)
                net.set_control_signals(control_signals)

                output = net(batch_data_god)

                control_loss = criterion(output, batch_labels_god)

                l1_reg = l1_reg = l1_lambda * sum(
                    p.abs().sum() for p in net.parameters()
                )

                total_control_loss = control_loss + l1_reg

                total_control_loss.backward()
                control_optimizer.step()

                if abs(old_loss - total_control_loss.item()) < control_threshold:
                    break

                old_loss = total_control_loss.item()

            if control_loss.item() > 0.01:
                batch_losses.append(control_loss.item())
                with torch.no_grad():
                    net.reset_control_signals()
                    h1 = net.layer1(net.flatten(batch_data_god))
                    output = net(batch_data_god)
                    current_activities = torch.cat(
                        [net.flatten(batch_data_god), h1, output], dim=1
                    )
                    control_signals = control_net(current_activities)
                    # a.shape is [batch_size, hidden_size + output_size]
                    # control_signals = control_signals[no_god_mask]
                    a1 = control_signals[:, : net.hidden_size]
                    a2 = control_signals[:, net.hidden_size :]

                    # Sander said, we can use 1.0 as the baseline
                    baseline_a1 = torch.ones_like(a1)
                    baseline_a2 = torch.ones_like(a2)
                    a1_diff = a1 - baseline_a1
                    a2_diff = a2 - baseline_a2

                    # Layer 1 weight update
                    x = net.flatten(batch_data_god)
                    phi = net.hidden_activations(net.layer1(x))
                    r_post_adjusted = phi * a1 * a1_diff
                    dw = r_post_adjusted.T @ x
                    dw = dw / x.shape[0]
                    net.layer1.weight.grad = torch.clamp(dw, min=-5, max=5)

                    # Layer 2 weight update
                    x2 = net.hidden_activations(net.layer1(net.flatten(batch_data_god)))
                    phi2 = net.output_activations(net.layer2(x2))
                    r_post_adjusted2 = phi2 * a2 * a2_diff
                    dw2 = r_post_adjusted2.T @ x2
                    dw2 = dw2 / x2.shape[0]
                    net.layer2.weight.grad = torch.clamp(dw2, min=-5, max=5)

                    net_optimizer.step()

        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        pbar.set_postfix(avg_epoch_loss=epoch_loss)


def run_experiment(params):
    from utils.random_conf import ensure_deterministic

    ensure_deterministic()
    num_epochs = params["num_epochs"]
    inner_epochs = params["inner_epochs"]
    learning_rate = params["learning_rate"]
    control_lr = params["control_lr"]
    control_threshold = params["control_threshold"]
    l1_lambda = params["l1_lambda"]

    # Size of all the "activities" from Net we use as input
    input_size_net = 784  # Flattened image: 28 x 28
    hidden_size_net = 100
    output_size_net = 10
    hidden_size_control = 150

    input_size_control = input_size_net + hidden_size_net + output_size_net

    net = Net(
        input_size=input_size_net,
        hidden_size=hidden_size_net,
        output_size=output_size_net,
        softmax=False,
    )

    control_net = ControlNet(
        input_size=input_size_control,
        hidden_size=hidden_size_control,
        output_size=hidden_size_net + output_size_net,
    )

    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    train_dataloaders = []
    test_dataloaders = []
    task_ids = list(range(1, 2))
    for task_id in task_ids:
        train_loader_god, test_loader_god = get_dataloaders(
            task_id=task_id,
            train_batch_size=128,
            test_batch_size=128,
        )

        train_dataloaders.append(train_loader_god)
        test_dataloaders.append(test_loader_god)

        train_model(
            task_id,
            net,
            control_net,
            train_loader_god,
            criterion,
            control_optimizer,
            net_optimizer,
            control_threshold,
            num_epochs,
            inner_epochs,
            l1_lambda,
        )

        acc = evaluate_model(net, control_net, test_loader_god, useSignals=True)
        return acc

        # for sub_task_id in range(task_id):
        #     acc_with = evaluate_model(
        #         net, control_net, test_dataloaders[sub_task_id], useSignals=True
        #     )
        #     acc_without = evaluate_model(
        #         net, control_net, test_dataloaders[sub_task_id], useSignals=False
        #     )
        #     if task_id == sub_task_id + 1:
        #         print_green(
        #             f"[with] Task {task_id} - {sub_task_id + 1}: {acc_with:.2f}%"
        #         )
        #         print_green(
        #             f"[without] Task {task_id} - {sub_task_id + 1}: {acc_without:.2f}%"
        #         )
        #     else:
        #         print(f"[with] Task {task_id} - {sub_task_id + 1}: {acc_with:.2f}%")
        #         print(
        #             f"[without] Task {task_id} - {sub_task_id + 1}: {acc_without:.2f}%"
        #         )


# Objective function for Optuna
def objective(trial, run_name):
    # Define the hyperparameter search space
    num_epochs = trial.suggest_int("num_epochs", 2, 4)
    inner_epochs = trial.suggest_int("inner_epochs", 10, 200)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)
    control_lr = trial.suggest_float("control_lr", 1e-8, 1e-1, log=True)
    control_threshold = trial.suggest_float("control_threshold", 1e-8, 1e-1, log=True)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-12, 10, log=True)

    # Run the model with the sampled parameters
    params = {
        "num_epochs": num_epochs,
        "inner_epochs": inner_epochs,
        "learning_rate": learning_rate,
        "control_lr": control_lr,
        "control_threshold": control_threshold,
        "l1_lambda": l1_lambda,
    }

    acc = run_experiment(params)

    # Goal is to maximize avg_accuracy
    return acc


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

    results_dir = os.path.join("results/tl_full_XOR", run_name)
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


run_optuna_study("test_minimalexample", 100, 8)
