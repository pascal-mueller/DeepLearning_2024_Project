import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import optuna
import sqlite3

from nn.Net import Net
from nn.ControlNet import ControlNet
from dataloaders.MNISTDataset import get_dataloaders
from utils.random_conf import ensure_deterministic
from utils.save_model_with_grads import save_model_with_grads
from utils.fisher_information_metric import plot_FIM
from utils.plot_losses import plot_losses as plot_losses_fn
from utils.plot_control_signals import plot_control_signals


BEST_PARAMS = {
    "num_epochs": 4,
    "inner_epochs": 156,
    "learning_rate": 3.3258829408007178e-06,
    "control_lr": 2.742808475368645e-05,
    "control_threshold": 1.103321094318002e-14,
    "l1_lambda": 1.725811981775536,
}


def evaluate_model(net, control_net, eval_loader, verbose_level=0):
    correct = 0
    total = 0

    with torch.no_grad():
        for eval_data, eval_labels in eval_loader:
            net.reset_control_signals()

            if verbose_level >= 2:
                print("Eval", eval_data.shape, eval_labels)

            h1 = net.layer1(net.flatten(eval_data))
            # out = net.layer2(net.hidden_activations(h1))
            output = net(eval_data)
            if verbose_level >= 2:
                print("Output", output)
            current_activities = torch.cat([net.flatten(eval_data), h1, output], dim=1)

            if verbose_level >= 2:
                print("Current activities", current_activities)

            control_signals = control_net(current_activities)

            # Question: Why do we need to set the control signals here?
            # Eval should be net only no?
            # net.set_control_signals(control_signals)

            if verbose_level >= 2:
                print("Control signals", net.hidden_activations.get_control_signals())

            outputs = net(eval_data)

            if verbose_level >= 2:
                print(outputs)

            predicted = outputs.max(1)[1]  # dim 1
            total += eval_labels.size(0)
            correct += (predicted == eval_labels).sum().item()

            if verbose_level >= 2:
                print(predicted)

    return 100 * correct / total


def train_model(
    net,
    control_net,
    train_loader,
    test_loader,
    task_id,
    num_epochs,
    inner_epochs,
    net_optimizer,
    control_optimizer,
    criterion,
    l1_lambda,
    control_threshold,
    device=torch.device("cpu"),
    verbose_level=0,
):
    print(f"Training on device {device}")

    pbar = tqdm(
        range(num_epochs),
        desc=f"Task {task_id} Epochs",
        leave=False,
        disable=(verbose_level < 0),
    )

    task_losses = []
    control_signal_history = []

    for epoch in pbar:
        epoch_losses = []
        control_signal_history_batch = []
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Get current network activities
            with torch.no_grad():
                net.reset_control_signals()
                h1 = net.layer1(net.flatten(batch_data))
                output = net(batch_data)
                current_activities = torch.cat(
                    [net.flatten(batch_data), h1, output], dim=1
                )

            if verbose_level >= 1:
                print(
                    f"Processing batch {batch_idx}/{len(train_loader)} of task {task_id}"
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

                # if verbose_level >= 2:
                #     print(f"Inner epoch {inner_epoch}")

                output = net(batch_data)  # net is excluded from the graph

                control_loss = criterion(output, batch_labels)

                # graph = make_dot(control_loss, params=params)
                # graph.render("combined_computational_graph", format="pdf")

                output_tmp = net(batch_data)
                rows = torch.arange(output_tmp.size(0))
                output_tmp[rows, batch_labels] -= 1

                l1_reg = l1_lambda * output_tmp.abs().sum(dim=1).mean()

                total_control_loss = control_loss + l1_reg

                # Note: This does BP over the connected graph of
                # net and control_net!
                total_control_loss.backward()

                # Note: This only updates the weights for control_net!
                control_optimizer.step()
                # print(control_signals)
                # print(control_net.layer1.weight.grad.sum())
                # input()

                if abs(prev_loss - total_control_loss.item()) < control_threshold:
                    if verbose_level >= 2:
                        print("  Converged at epoch", inner_epoch)
                    break

                prev_loss = total_control_loss.item()

                epoch_losses.append(control_loss.item())

            # Update weights based on control signals
            if total_control_loss.item() > 0.01:
                with torch.no_grad():
                    control_signals = control_net(current_activities)
                    control_signal_history_batch.append(control_signals)
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
                    # x = net.flatten(batch_data)

                    # # phi.shape is [batch_size, hidden_size]
                    # phi = net.hidden_activations(net.layer1(x))

                    # # # Loop over post-synaptic neurons (output neurons of layer 1)
                    # for i in range(net.hidden_size):
                    #     # The post synaptic neuron j has output phi_i
                    #     r_post = phi[:, i] * a1[:, i]  # r_post.shape is [batch_size]

                    #     # Loop over presynaptic signals (the input signals for the i-th post-synaptic neuron)
                    #     for j in range(net.input_size):
                    #         # Post synaptic neuron i has presynaptic signal j
                    #         r_pre_j = x[:, j]  # r_pre.shape is [batch_size]

                    #         dw_ij = (
                    #             r_pre_j * r_post * a1_diff[:, i]
                    #         )  # dw_i.shape is [batch_size]

                    #         # Note: We take the mean because we have a batch!
                    #         # Note: We set the gradient of the weight because later on
                    #         # we use an optimizer to update the weight.
                    #         net.layer1.weight.grad[i, j] = dw_ij.mean()

                    # 1) Flatten or otherwise prepare your input batch
                    x = net.flatten(batch_data)  # shape: [batch_size, input_size]

                    # 2) Compute the post-activation phi (already done by net.hidden_activations)
                    phi = net.hidden_activations(
                        net.layer1(x)
                    )  # shape: [batch_size, hidden_size]

                    # 3) Elementwise multiply to get the “adjusted” post-synaptic signal:
                    #    r_post_adjusted[n, i] = phi[n, i] * a1[n, i] * a1_diff[n, i]
                    r_post_adjusted = (
                        phi * a1 * a1_diff
                    )  # shape: [batch_size, hidden_size]

                    # 4) The outer product over the batch dimension:
                    #    result is [hidden_size, input_size]
                    dw = r_post_adjusted.T @ x

                    # 5) Take the mean over the batch
                    dw = dw / x.shape[0]

                    net.layer1.weight.grad = torch.clamp(dw, min=-1, max=1)
                    print(net.layer1.weight.grad.mean())

                    #
                    # LAYER 2 WEIGHT UPDATE
                    #
                    x = net.hidden_activations(
                        net.layer1(net.flatten(batch_data))
                    )  # x.shape is [batch_size, hidden_size]

                    phi = net.output_activations(
                        net.layer2(x)
                    )  # phi.shape is [batch_size, output_size]

                    # # Loop over post-synaptic neurons (output neurons of layer 2)
                    # for i in range(net.output_size):
                    #     # The post synaptic neuron j has output phi_i
                    #     r_post = phi[:, i] * a2[:, i]  # r_post.shape is [batch_size]

                    #     # Loop over presynaptic signals (the input signals for the i-th post-synaptic neuron)
                    #     for j in range(net.hidden_size):
                    #         # Post synaptic neuron i has presynaptic signal j
                    #         r_pre_i = x[:, j]  # r_pre.shape is [batch_size]

                    #         dw_i = (
                    #             r_pre_i * r_post * a2_diff[:, i]
                    #         )  # dw_i.shape is [batch_size]

                    #         # We take the mean because we have a batch!
                    #         net.layer2.weight.grad[i, j] = dw_i.mean()

                    r_post_adjusted = phi * a2 * a2_diff
                    dw = r_post_adjusted.T @ x
                    dw = dw / x.shape[0]
                    net.layer2.weight.grad = torch.clamp(dw, min=-1, max=1)

                    print(net.layer2.weight.grad.mean())

                    net_optimizer.step()

                    print(net.layer1.weight.mean())
                    print(net.layer2.weight.mean())

                    print("-----------------")

        # len(control_signal_history_batch) = num_batches
        # control_signal_history_batch[0].shape = [batch_size, hidden_size + output_size]
        #
        # [batch_size, hidden_size + output_size]
        # [batch_size, hidden_size + output_size]
        # [batch_size, hidden_size + output_size]
        # [batch_size, hidden_size + output_size]
        # [batch_size, hidden_size + output_size]
        # [batch_size, hidden_size + output_size]

        # [num_samples, hidden_size + output_size]
        signals_stacked = torch.cat(control_signal_history_batch)

        hidden_signal_history = signals_stacked[:, : net.hidden_size]
        output_signal_history = signals_stacked[:, net.hidden_size :]

        # Take the mean over all samples and over all hidden neurons
        hidden_signal_mean = torch.mean(hidden_signal_history)
        output_signal_mean = torch.mean(output_signal_history)

        control_signal_history.append([hidden_signal_mean, output_signal_mean])

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        task_losses.append(avg_epoch_loss)
        if epoch % 1 == 0 and verbose_level >= 0:
            pbar.set_postfix(avg_epoch_loss=avg_epoch_loss)
        accuracy = evaluate_model(net, control_net, test_loader, verbose_level=1)
        print(f"Epoch {epoch} Accuracy: {accuracy:.2f}%")

    plot_control_signals(
        control_signal_history, results_dir, filename=f"control_signals.png"
    )

    return task_losses


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
    test_run=False,
    device=torch.device("cpu"),
):
    (
        num_epochs,
        inner_epochs,
        learning_rate,
        control_lr,
        control_threshold,
        l1_lambda,
    ) = params.values()

    task_ids = range(1, 5)

    if test_run:
        task_ids = [0]

    results_dir = os.path.join("results", "tl_fmnist", run_name)
    os.makedirs(results_dir, exist_ok=True)

    net = Net(input_size=784, hidden_size=100, output_size=10, softmax=True).to(device)
    control_net = ControlNet(
        input_size=784 + 100 + 2 * 10, hidden_size=100, output_size=110
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    all_losses = []
    task_performance = {}

    dataloaders = {"train": [], "test": []}
    for task_id in task_ids:
        train_loader, test_loader = get_dataloaders(
            task_id, batch_size=512, device=device
        )
        dataloaders["train"].append(train_loader)
        dataloaders["test"].append(test_loader)

        if plot_data:
            raise NotImplementedError(
                "Plotting data is not implemented yet for tl fashiomnist."
            )
            # plot_subset_fn(
            #     train_loader.dataset,
            #     results_dir,
            #     filename=f"task_{task_id}_train_data.png",
            #     title=f"Task {task_id} Train Data",
            # )
            # plot_subset_fn(
            #     test_loader.dataset,
            #     results_dir=results_dir,
            #     filename=f"task_{task_id}_test_data.png",
            #     title=f"Task {task_id} Test Data",
            # )

    if plot_data:
        raise NotImplementedError(
            "Plotting data is not implemented yet for tl fashiomnist."
        )
    # plot_dataloaders(
    #     dataloaders["train"],
    #     results_dir,
    #     filename="train_data.png",
    #     title="Train Data",
    # )
    # plot_dataloaders(
    #     dataloaders["test"],
    #     results_dir,
    #     filename="test_data.png",
    #     title="Test Data",
    # )

    for task_id in task_ids:
        train_loader = dataloaders["train"][task_id - 1]
        test_loader = dataloaders["test"][task_id - 1]

        task_losses = train_model(
            net,
            control_net,
            train_loader,
            task_id,
            num_epochs,
            inner_epochs,
            net_optimizer,
            control_optimizer,
            criterion,
            l1_lambda,
            control_threshold,
            verbose_level=0,
        )
        all_losses.extend(task_losses)

        task_performance[task_id] = {}

        if not test_run:
            for eval_task_id in range(1, task_id + 1):
                test_loader = dataloaders["test"][eval_task_id - 1]
                accuracy = evaluate_model(net, control_net, test_loader, verbose_level)
                task_performance[task_id][eval_task_id] = accuracy
                if verbose_level >= 0:
                    print(
                        f"Task {task_id} - Performance on Task {eval_task_id}: {accuracy:.2f}%"
                    )
        else:
            accuracy = evaluate_model(net, control_net, test_loader, verbose_level)
            task_performance[0][0] = accuracy
            print(f"Accuracy on test set: {accuracy:.2f}%")

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
def objective(trial, run_name):
    # Define the hyperparameter search space
    num_epochs = trial.suggest_int("num_epochs", 15, 30)
    inner_epochs = trial.suggest_int("inner_epochs", 10, 200)
    learning_rate = trial.suggest_float("learning_rate", 1e-9, 1e-1, log=True)
    control_lr = trial.suggest_float("control_lr", 1e-8, 1e-1, log=True)
    control_threshold = trial.suggest_float("control_threshold", 1e-14, 1e-4, log=True)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-6, 4e-1, log=True)

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
        params, run_name=run_name, verbose_level=0
    )

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


if __name__ == "__main__":
    ensure_deterministic()
    torch.set_printoptions(threshold=100000)

    num_epochs = 50
    inner_epochs = 156
    learning_rate = 0.001
    control_lr = 0.001
    control_threshold = 1.103321094318002e-8
    l1_lambda = 0.0

    device = torch.device("cpu")

    results_dir = os.path.join("results", "tl_fmnist", "classical_full")
    os.makedirs(results_dir, exist_ok=True)

    input_size_net = 784  # Flattened image: 28 x 28
    hidden_size_net = 200
    output_size_net = 10
    hidden_size_control = 600

    # Size of all the "activities" from Net we use as input
    input_size_control = input_size_net + hidden_size_net + output_size_net

    net = Net(
        input_size=input_size_net,
        hidden_size=hidden_size_net,
        output_size=output_size_net,
        softmax=False,
    ).to(device)

    control_net = ControlNet(
        input_size=input_size_control,
        hidden_size=hidden_size_control,
        output_size=hidden_size_net + output_size_net,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    all_losses = []
    task_performance = {}

    task_id = 0

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    # TODO: If we take a subset, we should compute mean and std again!
    transform = transforms.Compose(
        [
            # Converts [H,W] PIL image in [0,255] -> FloatTensor [C,H,W] in [0,1]
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    trainset = datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )
    testset = datasets.MNIST(
        root="data", train=False, transform=transform, download=True
    )

    small_trainset = False

    if small_trainset:
        # Randomly select 10% of indices
        all_indices = np.arange(len(trainset))
        np.random.shuffle(all_indices)

        num_samples = int(len(trainset) * 0.1)
        subset_indices = all_indices[:num_samples]

        # Create a Subset
        trainset = Subset(trainset, subset_indices)

    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(testset, batch_size=512, shuffle=False)

    # train_loader, test_loader = get_dataloaders(task_id, batch_size=512)

    verbose_level = 0
    task_losses = train_model(
        net,
        control_net,
        train_loader,
        test_loader,
        task_id,
        num_epochs,
        inner_epochs,
        net_optimizer,
        control_optimizer,
        criterion,
        l1_lambda,
        control_threshold,
        device,
        verbose_level=verbose_level,
    )

    all_losses.extend(task_losses)

    accuracy = evaluate_model(
        net, control_net, test_loader, verbose_level=verbose_level
    )

    print(f"Accuracy on test set: {accuracy:.2f}%")
    print(all_losses)
    plot_losses_fn(all_losses, results_dir)
