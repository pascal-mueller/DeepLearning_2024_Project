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

from utils.colored_prints import *


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

            inp = net.flatten(batch_data)
            h1 = net.layer1(inp)
            h2 = net.layer2(net.h1_mrelu(h1))
            h3 = net.layer3(net.h2_mrelu(h2))
            output = net.layer4(net.h3_mrelu(h3))
            current_activities = torch.cat([inp, h1, h2, h3, output], dim=1)

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
    test_loader_god,
    criterion,
    control_optimizer,
    net_optimizer,
    control_threshold,
    num_epochs,
    inner_epochs,
    l1_lambda,
):
    pbar = tqdm(range(num_epochs), desc=f"Epochs", leave=False, disable=True)

    for epoch in pbar:
        # print(f"Press any key to start epoch {epoch}")
        # input()
        batch_losses = []

        for batch_id, (batch_data_god, batch_labels_god) in enumerate(train_loader_god):
            # Get current network activities
            with torch.no_grad():
                net.reset_control_signals()

                inp = net.flatten(batch_data_god)
                h1 = net.layer1(inp)
                h2 = net.layer2(net.h1_mrelu(h1))
                h3 = net.layer3(net.h2_mrelu(h2))
                output = net.layer4(net.h3_mrelu(h3))
                current_activities = torch.cat([inp, h1, h2, h3, output], dim=1)

            old_loss = float("inf")
            converged = False
            for inner_epoch in range(inner_epochs):
                control_optimizer.zero_grad()
                net_optimizer.zero_grad()  # TODO: Do I need this?

                control_signals = control_net(current_activities)
                net.set_control_signals(control_signals)

                output = net(batch_data_god)

                control_loss = criterion(output, batch_labels_god)

                l1_reg = 0.01 * sum(p.abs().sum() for p in control_net.parameters())
                # l2_lambda = 1e-2

                # l2_reg = l2_lambda * sum((p**2).sum() for p in control_net.parameters())
                # l1_reg = l1_lambda * sum(
                #     (output - 1).abs().sum() for output in net(batch_data_god)
                # )
                total_control_loss = control_loss + l1_reg

                total_control_loss.backward()
                control_optimizer.step()

                if abs(old_loss - total_control_loss.item()) < control_threshold:
                    converged = True
                    # print_info(f"Converged after {inner_epoch} epochs")
                    # if inner_epoch == 1:
                    #     breakpoint()
                    break

                old_loss = total_control_loss.item()

            if not converged:
                # print_error(f"Not converged")
                converged = False

            # print(f"{batch_id}/{len(train_loader_god)} ", total_control_loss.item())

            if control_loss.item() > 0.01:
                batch_losses.append(total_control_loss.item())
                with torch.no_grad():
                    # net.set_control_signals(control_signals)
                    net.reset_control_signals()
                    control_signals = control_net(current_activities)
                    # a.shape is [batch_size, hidden_size + output_size]
                    # control_signals = control_signals[no_god_mask]
                    a1 = control_signals[:, : net.hidden_size]
                    a2 = control_signals[:, net.hidden_size : 2 * net.hidden_size]
                    a3 = control_signals[:, 2 * net.hidden_size :]

                    # Sander said, we can use 1.0 as the baseline
                    baseline_a1 = torch.ones_like(a1)
                    baseline_a2 = torch.ones_like(a2)
                    baseline_a3 = torch.ones_like(a3)
                    a1_diff = a1 - baseline_a1
                    a2_diff = a2 - baseline_a2
                    a3_diff = a3 - baseline_a3

                    # Layer 1 weight update
                    x = net.flatten(batch_data_god)
                    phi = net.h1_mrelu(net.layer1(x))
                    r_post_adjusted = phi * a1_diff
                    dw = r_post_adjusted.T @ x
                    dw = dw / x.shape[0]
                    net.layer1.weight.grad = dw

                    # Layer 2 weight update
                    x2 = net.h1_mrelu(net.layer1(net.flatten(batch_data_god)))
                    phi2 = net.h2_mrelu(net.layer2(x2))

                    r_post_adjusted2 = phi2 * a2_diff
                    dw2 = r_post_adjusted2.T @ x2
                    dw2 = dw2 / x2.shape[0]
                    net.layer2.weight.grad = dw2

                    # Layer 3 weight update
                    x3 = net.h2_mrelu(
                        net.layer2(
                            net.h1_mrelu(net.layer1(net.flatten(batch_data_god)))
                        )
                    )
                    phi3 = net.h3_mrelu(net.layer3(x3))

                    r_post_adjusted3 = phi3 * a3_diff
                    dw3 = r_post_adjusted3.T @ x3
                    dw3 = dw3 / x3.shape[0]
                    net.layer3.weight.grad = dw3

                    # print(dw.mean(), dw2.mean(), dw3.mean())

                net_optimizer.step()

        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        pbar.set_postfix(avg_epoch_loss=epoch_loss)

        acc = evaluate_model(net, control_net, test_loader_god, useSignals=True)

        # print(f"Task {task_id} - Epoch {epoch}: {acc:.2f}%")
        # if acc > 80:
        #     print("Early stopping")
        #     break


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

    hidden_size_control = 100
    output_size_control = 3 * hidden_size_net

    input_size_control = input_size_net + 3 * hidden_size_net + output_size_net

    net = Net(
        input_size=input_size_net,
        hidden_size=hidden_size_net,
        output_size=output_size_net,
        softmax=False,
    )

    control_net = ControlNet(
        input_size=input_size_control,
        hidden_size=hidden_size_control,
        output_size=output_size_control,
    )

    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    # )

    # train_dataset_full = datasets.MNIST(
    #     root="local_data", train=True, transform=transform, download=True
    # )
    # test_dataset_full = datasets.MNIST(
    #     root="local_data", train=False, transform=transform, download=True
    # )

    # train_loader = DataLoader(train_dataset_full, batch_size=1024, shuffle=True)
    # test_loader = DataLoader(test_dataset_full, batch_size=512, shuffle=False)

    # train_dataloaders = []
    # test_dataloaders = []
    # task_ids = list(range(0, 1))
    # for task_id in task_ids:
    train_dataloaders, test_dataloaders = get_dataloaders(
        train_batch_size=128,
        test_batch_size=128,
    )

    # train_dataloaders.append(train_loader_god)
    # test_dataloaders.append(test_loader_god)

    train_model(
        0,
        net,
        control_net,
        train_dataloaders[0],
        test_dataloaders[0],
        criterion,
        control_optimizer,
        net_optimizer,
        control_threshold,
        num_epochs,
        inner_epochs,
        l1_lambda,
    )

    for task_id in range(1, 5):
        if task_id == 0:
            print_info("Training on all data")
        else:
            print_info(f"Training on Task {task_id}")

        acc_train = evaluate_model(
            net, control_net, train_dataloaders[task_id], useSignals=True
        )
        acc_test = evaluate_model(
            net, control_net, test_dataloaders[task_id], useSignals=True
        )

        print(
            f"Task {task_id} - Train Acc: {acc_train:.2f}% - Test Acc: {acc_test:.2f}%"
        )

        for sub_task_id in range(1, task_id + 1):
            acc_with = evaluate_model(
                net, control_net, test_dataloaders[sub_task_id], useSignals=True
            )
            acc_without = evaluate_model(
                net, control_net, test_dataloaders[sub_task_id], useSignals=False
            )

            if task_id == sub_task_id:
                print_green(f"[with] Task {task_id} - {sub_task_id}: {acc_with:.2f}%")
                print_green(
                    f"[without] Task {task_id} - {sub_task_id}: {acc_without:.2f}%"
                )
            else:
                print(f"[with] Task {task_id} - {sub_task_id}: {acc_with:.2f}%")
                print(f"[without] Task {task_id} - {sub_task_id}: {acc_without:.2f}%")
        print("\n")


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


# params = {
#     "num_epochs": 3,
#     "inner_epochs": 188,
#     "learning_rate": 0.0001,
#     "control_lr": 0.0001,
#     "control_threshold": 0.00032949549118669864,
#     "l1_lambda": 0.0,
# }
params = {
    "num_epochs": 25,
    "inner_epochs": 200,
    "learning_rate": 1.651703048219e-05,
    "control_lr": 1.18078126401598e-05,
    "control_threshold": 0.00603542302442579,
    "l1_lambda": 0.000141872888572121,
}


run_experiment(params)
# run_optuna_study("test_minimalexample", 100, 8)
