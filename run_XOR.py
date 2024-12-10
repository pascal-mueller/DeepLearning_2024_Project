import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from networks import *
from plot import *
from data import *

# Fix seeds
torch.manual_seed(0)
np.random.seed(0)


def evaluate_model(net, control_net, eval_loader, verbose_level=0):
    correct = 0
    total = 0
    with torch.no_grad():
        for eval_data, eval_labels, _ in eval_loader:
            net.reset_control_signals()
            if verbose_level >= 2:
                print("Eval", eval_data.shape, eval_labels)
            h1 = net.layer1(net.flatten(eval_data))
            output = net(eval_data)
            if verbose_level >= 2:
                print("Output", output)
            current_activities = torch.cat([net.flatten(eval_data), h1, output], dim=1)

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


net = Net()
control_net = ControlNet()

num_epochs = 15
inner_epochs = 5
learning_rate = 0.0001
control_lr = 0.0001
control_threshold = 1e-8
l1_lambda = 0.1

criterion = nn.CrossEntropyLoss()
control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
net_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

all_losses = []
task_performance = {}

# 0: Print performance
# 1: Print loss per epoch
# 2: Print control signals, activities, outputs
verbose_level = 0
plot_data = False

# Loop over continuous learning tasks
for task_id in range(1, 4):
    print(f"Training on Task {task_id}")
    dataloader = get_dataloader(task_id)

    if plot_data:
        dataloader.dataset.plot()

    task_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_data, batch_labels, _ in dataloader:
            # TODO: What exactly is happening here?
            # Is it just a "forward pass"?
            # Get current network activities
            with torch.no_grad():
                # TODO: Understand this.
                net.reset_control_signals()
                # TODO: Why do we call h1? (I think it's to compute the
                # control signals but that's more a guess than knowledge)
                h1 = net.layer1(net.flatten(batch_data))

                # "Forward pass"
                output = net(batch_data)

                # TODO: Why net.flatten? Does nn.Flatten() contain hyper
                # parameteres or something like that?
                current_activities = torch.cat(
                    [net.flatten(batch_data), h1, output], dim=1
                )

            # Inner loop - Training the control network
            prev_loss = float("inf")
            for inner_epoch in range(inner_epochs):
                control_optimizer.zero_grad()

                control_signals = control_net(current_activities)
                if verbose_level >= 2:
                    print("Control signals", control_signals.mean())
                net.set_control_signals(control_signals)

                output = net(batch_data)
                control_loss = criterion(output, batch_labels)
                # Note: Fucking list comprehension
                l1_reg = l1_lambda * sum(
                    (output - 1).abs().sum() for output in net(batch_data)
                )
                total_control_loss = control_loss + l1_reg

                total_control_loss.backward()
                control_optimizer.step()

                if abs(prev_loss - total_control_loss.item()) < control_threshold:
                    if verbose_level >= 1:
                        print("  Converged at epoch", inner_epoch)
                    break

                prev_loss = total_control_loss.item()

            # # Update weights based on control signals
            # if total_control_loss.item() > 0.01:
            #     with torch.no_grad():
            #         pass
            #         # Get pre-synaptic activities
            #         # TODO
            #         # Calculate weight updates using the control-based rule
            #         # Layer 1 updates
            #         # TODO
            #         # Layer 2 updates
            #         # TODO

            #     epoch_losses.append(control_loss.item())
            # Update weights based on control signals
            if total_control_loss.item() > 0.01:
                with torch.no_grad():
                    # Layer 1: Get pre-synaptic activities (flattened inputs)
                    pre_activities_layer1 = net.flatten(batch_data)

                    # Get the target signal (difference between output and label)
                    target_signal = batch_labels - output  # [batch_size, output_size]

                    # Modulate target signal by control signals
                    modulated_target_layer1 = control_signals[:, : net.hidden_size]
                    weight_update_layer1 = torch.einsum(
                        "bi,bj->ij", pre_activities_layer1, modulated_target_layer1
                    )

                    # Update weights for Layer 1
                    net.layer1.weight += learning_rate * weight_update_layer1.T

                    # Layer 2: Pre-synaptic activities (hidden activations after modulation)
                    pre_activities_layer2 = net.hidden_activations(
                        net.layer1(pre_activities_layer1)
                    )

                    # Modulate target signal for layer 2
                    modulated_target_layer2 = control_signals[:, net.hidden_size :]
                    weight_update_layer2 = torch.einsum(
                        "bi,bj->ij", pre_activities_layer2, modulated_target_layer2
                    )

                    # Update weights for Layer 2
                    net.layer2.weight += learning_rate * weight_update_layer2.T

                    net_optimizer.step()

                epoch_losses.append(control_loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        task_losses.append(avg_epoch_loss)
        if epoch % 1 == 0 and verbose_level >= 1:
            print(f" Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

    all_losses.extend(task_losses)

    # Evaluation remains the same
    task_performance[task_id] = {}
    for eval_task_id in range(1, task_id + 1):
        eval_loader = get_dataloader(eval_task_id)
        accuracy = evaluate_model(net, control_net, eval_loader, verbose_level)
        task_performance[task_id][eval_task_id] = accuracy
        print(f"Task {task_id} - Performance on Task {eval_task_id}: {accuracy:.2f}%")
