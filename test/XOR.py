import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from networks import *
from plot import *
from data import *


def evaluate_model(net, control_net, eval_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for eval_data, eval_labels, _ in eval_loader:
            net.reset_control_signals()
            # print("Eval", eval_data.shape, eval_labels)
            h1 = net.layer1(net.flatten(eval_data))
            output = net(eval_data)
            # print("Output", output)
            current_activities = torch.cat([net.flatten(eval_data), h1, output], dim=1)

            # print("Current activities", current_activities)

            control_signals = control_net(current_activities)
            net.set_control_signals(control_signals)
            # print("Control signals", net.hidden_activations.get_control_signals())
            outputs = net(eval_data)
            # print(outputs)

            predicted = outputs.max(1)[1]  # dim 1
            total += eval_labels.size(0)
            correct += (predicted == eval_labels.max(1)[1]).sum().item()
            # print(predicted)

    return 100 * correct / total


net = Net()
control_net = ControlNet()

num_epochs = 15
inner_epochs = 50
learning_rate = 0.1
control_lr = 0.01
control_threshold = 1e-3
l1_lambda = 0.01

criterion = nn.CrossEntropyLoss()
control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

all_losses = []
task_performance = {}

for task_id in range(1, 4):
    print(f"Training on Task {task_id}")
    dataloader = get_dataloader(task_id)
    task_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_data, batch_labels, _ in dataloader:
            # Get current network activities
            with torch.no_grad():
                net.reset_control_signals()
                h1 = net.layer1(net.flatten(batch_data))
                output = net(batch_data)
                current_activities = torch.cat(
                    [net.flatten(batch_data), h1, output], dim=1
                )

            # Inner loop - Training the control network
            prev_loss = float("inf")
            for inner_epoch in range(inner_epochs):
                control_optimizer.zero_grad()

                control_signals = control_net(current_activities)
                # print("Control signals", control_signals.mean())
                net.set_control_signals(control_signals)

                output = net(batch_data)
                control_loss = criterion(output, batch_labels)
                l1_reg = l1_lambda * sum(
                    (output - 1).abs().sum() for output in net(batch_data)
                )
                total_control_loss = control_loss + l1_reg

                total_control_loss.backward()
                control_optimizer.step()

                if abs(prev_loss - total_control_loss.item()) < control_threshold:
                    # print("Converged at epoch", inner_epoch)
                    break

                prev_loss = total_control_loss.item()

            # Update weights based on control signals
            if total_control_loss.item() > 0.01:
                with torch.no_grad():
                    # Get pre-synaptic activities
                    # TODO
                    # Calculate weight updates using the control-based rule
                    # Layer 1 updates
                    # TODO
                    # Layer 2 updates
                    # TODO
                    # a.shape is [batch_size, hidden_size + output_size]
                    a1 = control_signals[:, : net.hidden_size]
                    a2 = control_signals[:, net.hidden_size :]

                    # Sander said, we can use 1.0 as the baseline
                    baseline_a1 = torch.ones_like(a1)
                    baseline_a2 = torch.ones_like(a2)
                    a1_diff = a1 - baseline_a1
                    a2_diff = a2 - baseline_a2

                    x = net.flatten(batch_data)
                    phi = net.hidden_activations(net.layer1(x))
                    r_post_adjusted = phi * a1 * a1_diff
                    dw = r_post_adjusted.T @ x
                    dw1 = dw / x.shape[0]

                    net.layer1.weight += learning_rate * dw1

                    #
                    # LAYER 2 WEIGHT UPDATE
                    #
                    x = net.hidden_activations(net.layer1(net.flatten(batch_data)))

                    phi = net.output_activations(net.layer2(x))

                    r_post_adjusted = phi * a2 * a2_diff
                    dw = r_post_adjusted.T @ x
                    dw2 = dw / x.shape[0]
                    net.layer2.weight += learning_rate * dw2

                    # print("Layer 1", dw1)
                    # print("Layer 2", dw2)

                    # net_optimizer.step()

                epoch_losses.append(control_loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        task_losses.append(avg_epoch_loss)
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

    all_losses.extend(task_losses)

    # Evaluation remains the same
    task_performance[task_id] = {}
    for eval_task_id in range(1, task_id + 1):
        eval_loader = get_dataloader(eval_task_id)
        accuracy = evaluate_model(net, control_net, eval_loader)
        task_performance[task_id][eval_task_id] = accuracy
        print(f"Task {task_id} - Performance on Task {eval_task_id}: {accuracy:.2f}%")
