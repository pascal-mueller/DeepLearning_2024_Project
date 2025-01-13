import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import optuna
import random
import sqlite3
from torchviz import make_dot
from nn.Net import Net
from nn.ControlNet import ControlNet
from dataloaders.XORDataset import get_dataloaders
from utils.save_model_with_grads import save_model_with_grads
from utils.fisher_information_metric import plot_FIM
from utils.plot_losses import plot_losses as plot_losses_fn
from utils.plot_subset import plot_subset as plot_subset_fn
from utils.plot_data import plot_dataloaders

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

BEST_PARAMS = {
    "num_epochs": 30,
    "inner_epochs": 54,
    "learning_rate": 3.900671053825562e-06,
    "control_lr": 0.0008621989600943697,
    "control_threshold": 1.3565492056080836e-08,
    "l1_lambda": 0.0011869059296583477,
}


def evaluate_model(net, control_net, eval_loader, verbose_level=0):
    correct = 0
    total = 0
    with torch.no_grad():
        for eval_data, eval_labels, _ in eval_loader:
            net.reset_control_signals()
            if verbose_level >= 2:
                print("Eval", eval_data.shape, eval_labels)
            h1 = net.layer1(net.flatten(eval_data))
            out = net.layer2(net.hidden_activations(h1))
            output = net(eval_data)
            if verbose_level >= 2:
                print("Output", output)
            current_activities = torch.cat(
                [net.flatten(eval_data), h1, out, output], dim=1
            )

            if verbose_level >= 2:
                print("Current activities", current_activities)

            control_signals = control_net(current_activities)
            net.set_control_signals(control_signals)
            if verbose_level >= 2:
                print("Control signals", net.hidden_activations.get_control_signals())
            outputs = net(eval_data)

            if verbose_level >= 2:
                print(outputs)

            predicted = outputs.max(1)[1]  # dim 1
            total += eval_labels.size(0)
            correct += (predicted == eval_labels.max(1)[1]).sum().item()

            if verbose_level >= 2:
                print(predicted)

    return 100 * correct / total


def avg(data):
    values = [value for subdict in data.values() for value in subdict.values()]

    return sum(values) / len(values)


def run_experiment(
    params,
    run_name,
    verbose_level=-1,
    seed=0,
    plot_data=False,
    plot_losses=False,
    plot_fim=False,
):
    (
        num_epochs,
        inner_epochs,
        learning_rate,
        control_lr,
        control_threshold,
        l1_lambda,
    ) = params.values()

    results_dir = os.path.join("results", "tl_full_XOR", run_name)
    os.makedirs(results_dir, exist_ok=True)

    # Fix seeds
    # TODO: Make sure we fixed all possible randomness
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    net = Net()
    control_net = ControlNet()
    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    all_losses = []
    task_performance = {}

    dataloaders = {"train": [], "test": []}
    for task_id in range(1, 4):
        train_loader, test_loader = get_dataloaders(task_id)
        dataloaders["train"].append(train_loader)
        dataloaders["test"].append(test_loader)

        if plot_data:
            plot_subset_fn(
                train_loader.dataset,
                results_dir,
                filename=f"task_{task_id}_train_data.png",
                title=f"Task {task_id} Train Data",
            )
            plot_subset_fn(
                test_loader.dataset,
                results_dir=results_dir,
                filename=f"task_{task_id}_test_data.png",
                title=f"Task {task_id} Test Data",
            )

    if plot_data:
        plot_dataloaders(
            dataloaders["train"],
            results_dir,
            filename="train_data.png",
            title="Train Data",
        )
        plot_dataloaders(
            dataloaders["test"],
            results_dir,
            filename="test_data.png",
            title="Test Data",
        )

    for task_id in range(1, 4):
        train_loader = dataloaders["train"][task_id - 1]
        test_loader = dataloaders["test"][task_id - 1]

        task_losses = []

        pbar = tqdm(
            range(num_epochs),
            desc=f"Task {task_id} Epochs",
            leave=False,
            disable=(verbose_level <= 0),
        )

        for epoch in pbar:
            epoch_losses = []
            for batch_data, batch_labels, _ in train_loader:
                # Get current network activities
                with torch.no_grad():
                    net.reset_control_signals()
                    h1 = net.layer1(net.flatten(batch_data))
                    output = net(batch_data)
                    out = net.layer2(net.hidden_activations(h1))
                    current_activities = torch.cat(
                        [net.flatten(batch_data), h1, out, output], dim=1
                    )

                # Inner loop - Training the control network
                prev_loss = float("inf")
                for inner_epoch in tqdm(
                    range(inner_epochs),
                    desc="Inner Epochs",
                    leave=False,
                    disable=(True or verbose_level <= 0),
                ):
                    control_optimizer.zero_grad()

                    # Question: Why do we get the target signal for these
                    # activities? Shouldn't he take the target signal wtt to the
                    # inputs of ReLu layers?
                    control_signals = control_net(current_activities)

                    net.set_control_signals(control_signals)

                    # params = {
                    #     **dict(net.named_parameters()),
                    #     **dict(control_net.named_parameters()),
                    # }

                    if verbose_level >= 2:
                        print("Control signals", control_signals.mean())

                    output = net(batch_data)  # net is excluded from the graph
                    control_loss = criterion(output, batch_labels)
                    # graph = make_dot(control_loss, params=params)
                    # graph.render("combined_computational_graph", format="pdf")
                    l1_reg = (
                        l1_lambda
                        * (net(batch_data) - batch_labels).abs().sum(dim=1).mean()
                    )

                    total_control_loss = control_loss + l1_reg

                    # Note: This does BP over the connected graph of
                    # net and control_net!
                    total_control_loss.backward()

                    # Note: This only updates the weights for control_net!
                    control_optimizer.step()

                    if abs(prev_loss - total_control_loss.item()) < control_threshold:
                        if verbose_level >= 2:
                            print("  Converged at epoch", inner_epoch)
                        break

                    prev_loss = total_control_loss.item()

                    epoch_losses.append(control_loss.item())

                # Update weights based on control signals
                if total_control_loss.item() > 0.01:
                    with torch.no_grad():
                        # a.shape is [batch_size, hidden_size + output_size]
                        a1 = control_signals[:, : net.hidden_size]
                        a2 = control_signals[:, net.hidden_size :]

                        # Sander said, we can use 1.0 as the baseline
                        baseline_a1 = torch.ones_like(a1)
                        baseline_a2 = torch.ones_like(a2)
                        a1_diff = a1 - baseline_a1
                        a2_diff = a2 - baseline_a2

                        #
                        # LAYER 1 WEIGHT UPDATE
                        #

                        # x.shape is [batch_size, input_size]
                        x = net.flatten(batch_data)

                        # phi.shape is [batch_size, hidden_size]
                        phi = net.hidden_activations(net.layer1(x))

                        # Loop over post-synaptic neurons (output neurons of layer 1)
                        for i in range(net.hidden_size):
                            # The post synaptic neuron j has output phi_i
                            r_post = (
                                phi[:, i] * a1[:, i]
                            )  # r_post.shape is [batch_size]

                            # Loop over presynaptic signals (the input signals for the i-th post-synaptic neuron)
                            for j in range(net.input_size):
                                # Post synaptic neuron i has presynaptic signal j
                                r_pre_j = x[:, j]  # r_pre.shape is [batch_size]

                                dw_ij = (
                                    r_pre_j * r_post * a1_diff[:, i]
                                )  # dw_i.shape is [batch_size]

                                # Note: We take the mean because we have a batch!
                                # Note: We set the gradient of the weight because later on
                                # we use an optimizer to update the weight.
                                net.layer1.weight.grad[i, j] = dw_ij.mean()

                        #
                        # LAYER 2 WEIGHT UPDATE
                        #
                        x = net.hidden_activations(
                            net.layer1(net.flatten(batch_data))
                        )  # x.shape is [batch_size, hidden_size]

                        phi = net.output_activations(
                            net.layer2(x)
                        )  # phi.shape is [batch_size, output_size]

                        # Loop over post-synaptic neurons (output neurons of layer 2)
                        for i in range(net.output_size):
                            # The post synaptic neuron j has output phi_i
                            r_post = (
                                phi[:, i] * a2[:, i]
                            )  # r_post.shape is [batch_size]

                            # Loop over presynaptic signals (the input signals for the i-th post-synaptic neuron)
                            for j in range(net.hidden_size):
                                # Post synaptic neuron i has presynaptic signal j
                                r_pre_i = x[:, j]  # r_pre.shape is [batch_size]

                                dw_i = (
                                    r_pre_i * r_post * a2_diff[:, i]
                                )  # dw_i.shape is [batch_size]

                                # We take the mean because we have a batch!
                                net.layer2.weight.grad[i, j] = dw_i.mean()

                        net_optimizer.step()

            avg_epoch_loss = (
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            )
            task_losses.append(avg_epoch_loss)
            if epoch % 1 == 0 and verbose_level >= 1:
                pbar.set_postfix(avg_epoch_loss=avg_epoch_loss)

        all_losses.extend(task_losses)

        task_performance[task_id] = {}
        for eval_task_id in range(1, task_id + 1):
            test_loader = dataloaders["test"][eval_task_id - 1]
            accuracy = evaluate_model(net, control_net, test_loader, verbose_level)
            task_performance[task_id][eval_task_id] = accuracy
            if verbose_level >= 0:
                print(
                    f"Task {task_id} - Performance on Task {eval_task_id}: {accuracy:.2f}%"
                )

        # Save model
        # TODO: save optimizer state, epoch, loss and create a loading function
        save_model_with_grads(net, os.path.join(results_dir, f"net_task_{task_id}.pt"))
        save_model_with_grads(
            control_net, os.path.join(results_dir, f"control_net_task_{task_id}.pt")
        )

        # Save data
        torch.save(
            dataloaders["train"][task_id - 1].dataset,
            os.path.join(results_dir, f"task_{task_id}_data.pt"),
        )

    if plot_fim:
        plot_FIM(net, control_net, dataloaders)

    if plot_losses:
        plot_losses_fn(all_losses, results_dir)

    avg_perf = avg(task_performance)

    return params, task_performance, avg_perf


# Objective function for Optuna
def objective(trial):
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
    _, task_performance, avg_perf = run_experiment(params, verbose_level=-1)

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
    num_trials, num_cpus, dbname="paramsearch", study_name="hyperparameter_optimization"
):
    assert num_cpus <= 48, "Max 48 CPUs supported for SQLite storage"

    results_dir = "results"
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

    # Run optimization with parallel trials
    study.optimize(objective, n_trials=num_trials, n_jobs=num_cpus)

    # Print and return the best results
    print("Best parameters:", study.best_params)
    print("Best value (accuracy):", study.best_value)
    print("Best trial:", study.best_trial)
    return study


if __name__ == "__main__":
    print("Running full XOR experiment...")

    # You can run the XOR experiment with a specifi set of hyperparams:
    _, perf = run_experiment(BEST_PARAMS, verbose_level=1, plot_data=False)
    avg_perf = avg(perf)
    print("Avg Perf.: ", avg_perf)

    # # Remove
    quit()

    # Or you can start a hyperparameter optimization study with Optuna:
    # Configure the number of trials and CPUs
    num_trials = 1000
    # Adjust as needed, max is 48 (change storage string abvoe to increase max)
    num_cpus = 8

    # Run the study
    study = run_optuna_study(num_trials, num_cpus)

    # Show the best results
    print("Finished optimization. Best parameters:")
    print(study.best_params)
