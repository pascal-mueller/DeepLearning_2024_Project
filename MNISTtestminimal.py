import torch
import torch.nn as nn
from tqdm import tqdm
from nn.Net import Net
from nn.ControlNet import ControlNet
from utils.colored_prints import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    net,
    control_net,
    train_loader,
    criterion,
    control_optimizer,
    net_optimizer,
    control_threshold,
    num_epochs,
    inner_epochs,
    l1_lambda,
):
    pbar = tqdm(range(num_epochs), desc=f"Epochs", leave=False)

    for epoch in pbar:
        # print(f"Press any key to start epoch {epoch}")
        # input()
        batch_losses = []

        for batch_data, batch_labels in train_loader:
            # Get current network activities
            with torch.no_grad():
                net.reset_control_signals()
                h1 = net.layer1(net.flatten(batch_data))
                output = net(batch_data)
                current_activities = torch.cat(
                    [net.flatten(batch_data), h1, output], dim=1
                )

            old_loss = float("inf")
            for inner_epoch in range(inner_epochs):
                control_optimizer.zero_grad()
                net_optimizer.zero_grad()  # TODO: Do I need this?

                control_signals = control_net(current_activities)
                net.set_control_signals(control_signals)

                output = net(batch_data)

                control_loss = criterion(output, batch_labels)

                l1_reg = l1_reg = l1_lambda * sum(
                    p.abs().sum() for p in net.parameters()
                )

                total_control_loss = control_loss + l1_reg

                total_control_loss.backward()

                control_optimizer.step()
                net_optimizer.step()

                if abs(old_loss - total_control_loss.item()) < control_threshold:
                    break

                old_loss = total_control_loss.item()

            if control_loss.item() > 0.01:
                batch_losses.append(control_loss.item())
                with torch.no_grad():
                    control_signals = control_net(current_activities)
                    # a.shape is [batch_size, hidden_size + output_size]
                    a1 = control_signals[:, : net.hidden_size]
                    a2 = control_signals[:, net.hidden_size :]

                    # Sander said, we can use 1.0 as the baseline
                    baseline_a1 = torch.ones_like(a1)
                    baseline_a2 = torch.ones_like(a2)
                    a1_diff = a1 - baseline_a1
                    a2_diff = a2 - baseline_a2

                    # Layer 1 weight update
                    x = net.flatten(batch_data)
                    phi = net.hidden_activations(net.layer1(x))
                    r_post_adjusted = phi * a1 * a1_diff
                    dw = r_post_adjusted.T @ x
                    dw = dw / x.shape[0]
                    net.layer1.weight.grad = torch.clamp(dw, min=-1, max=1)

                    # Layer 2 weight update
                    x2 = net.hidden_activations(net.layer1(net.flatten(batch_data)))
                    phi2 = net.output_activations(net.layer2(x2))
                    r_post_adjusted2 = phi2 * a2 * a2_diff
                    dw2 = r_post_adjusted2.T @ x2
                    dw2 = dw2 / x2.shape[0]
                    net.layer2.weight.grad = torch.clamp(dw2, min=-1, max=1)

                    net_optimizer.step()

        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        pbar.set_postfix(avg_epoch_loss=epoch_loss)


def run_experiment():
    from dataloaders.MNISTDataset import get_dataloaders

    num_epochs = 15
    inner_epochs = 150
    learning_rate = 0.001
    control_lr = 0.01
    control_threshold = 1e-3
    l1_lambda = 0.01

    # Size of all the "activities" from Net we use as input
    input_size_net = 784  # Flattened image: 28 x 28
    hidden_size_net = 150
    output_size_net = 10
    hidden_size_control = 1000

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

    # def foo(module, grad_input, grad_output):
    #     if grad_input[0] is not None and grad_input[0].mean().item() == 0.0:
    #         breakpoint()

    #     if grad_output[0] is not None and grad_output[0].mean().item() == 0.0:
    #         breakpoint()

    # control_net.layer1.register_full_backward_hook(foo)
    # control_net.layer2.register_full_backward_hook(foo)

    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    train_dataloaders = []
    test_dataloaders = []
    for task_id in range(1, 5):
        train_loader, test_loader = get_dataloaders(
            task_id, train_batch_size=256, test_batch_size=128
        )
        train_dataloaders.append(train_loader)
        test_dataloaders.append(test_loader)

        train_model(
            net,
            control_net,
            train_loader,
            criterion,
            control_optimizer,
            net_optimizer,
            control_threshold,
            num_epochs,
            inner_epochs,
            l1_lambda,
        )

        for sub_task_id in range(task_id):
            acc_with = evaluate_model(
                net, control_net, test_dataloaders[sub_task_id], useSignals=True
            )
            acc_without = evaluate_model(
                net, control_net, test_dataloaders[sub_task_id], useSignals=False
            )
            if task_id == sub_task_id + 1:
                print_green(
                    f"[with] Task {task_id} - {sub_task_id + 1}: {acc_with:.2f}%"
                )
                print_green(
                    f"[without] Task {task_id} - {sub_task_id + 1}: {acc_without:.2f}%"
                )
            else:
                print(f"[with] Task {task_id} - {sub_task_id + 1}: {acc_with:.2f}%")
                print(
                    f"[without] Task {task_id} - {sub_task_id + 1}: {acc_without:.2f}%"
                )


run_experiment()
