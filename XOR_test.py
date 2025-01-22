import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
from nn.Net import Net
from nn.ControlNet import ControlNet
from utils.colored_prints import *

from HelpTensor import HelpTensor


data = torch.tensor(
    [[[0.9645, -0.0271], [-0.1611, 0.9617], [0.5719, 0.5491], [0.2849, 0.8134]]],
    requires_grad=True,
)

label = torch.tensor([0.0, 1.0])


# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

seed = 13456
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

num_epochs = 50
inner_epochs = 156
learning_rate = 0.001
control_lr = 0.001
control_threshold = 1.103321094318002e-8
l1_lambda = 0.0


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)

# Size of all the "activities" from Net we use as input
input_size_net = 8  # Flattened image: 28 x 28
hidden_size_net = 1
output_size_net = 2
hidden_size_control = 1

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

all_losses = []
task_performance = {}


task_losses = []

pbar = tqdm(
    range(num_epochs),
    desc=f"Epochs",
    leave=False,
)

for epoch in pbar:
    epoch_losses = []
    # Get current network activities
    with torch.no_grad():
        net.reset_control_signals()
        h1 = net.layer1(net.flatten(data))
        output = net(data)
        current_activities = torch.cat([net.flatten(data), h1, output], dim=1)

    breakpoint()
    # Inner loop - Training the control network
    total_control_loss_old = float("inf")
    for inner_epoch in tqdm(
        range(inner_epochs),
        desc="Inner Epochs",
        leave=False,
    ):
        control_optimizer.zero_grad()

        control_signals = control_net(current_activities)
        net.set_control_signals(control_signals)

        output = net(data)  # net is excluded from the graph

        from torchviz import make_dot

        dot = make_dot(output, params=dict(net.named_parameters()))
        dot.render("model_graph", format="png")

        # hardcoded label
        label_as_index = torch.argmax(label).unsqueeze(0)
        control_loss = criterion(output, label_as_index)

        # l1_reg = l1_lambda * (net(data) - label).abs().sum(dim=1).mean()

        grad_out = HelpTensor(torch.ones_like(control_loss))
        control_loss.backward(gradient=grad_out)
        breakpoint()
        control_optimizer.step()

        # print("\n")
        # print_info(f"Epoch {epoch} - Inner Epoch {inner_epoch}")
        # print(f"  Control Signals: {control_signals}")
        # print(f"  Output: {output}")
        # print(f"  Control Loss: {control_loss.item()}")
        # print(f"  control_net.layer1.weight.grad: {control_net.layer1.weight.grad}")
        # print(f"  control_net.layer2.weight.grad: {control_net.layer2.weight.grad}")
        # print(f"  control_net.layer1.weight: {control_net.layer1.weight}")
        # print(f"  control_net.layer2.weight: {control_net.layer2.weight}")
        # print(f"  net.layer1.weight.grad: {net.layer1.weight.grad}")
        # print(f"  net.layer2.weight.grad: {net.layer2.weight.grad}")
        # print(f"  net.layer1.weight: {net.layer1.weight}")
        # print(f"  net.layer2.weight: {net.layer2.weight}")

        # print(f"  L1 Reg: {l1_reg.item()}")
        # print(f"  Total Control Loss: {total_control_loss.item()}")

        # total_control_loss_old = total_control_loss.item()
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
            x = net.flatten(data)

            # phi.shape is [batch_size, hidden_size]
            phi = net.hidden_activations(net.layer1(x))

            # Loop over post-synaptic neurons (output neurons of layer 1)
            for i in range(net.hidden_size):
                # The post synaptic neuron j has output phi_i
                r_post = phi[:, i] * a1[:, i]  # r_post.shape is [batch_size]

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
                net.layer1(net.flatten(data))
            )  # x.shape is [batch_size, hidden_size]

            phi = net.output_activations(
                net.layer2(x)
            )  # phi.shape is [batch_size, output_size]

            # Loop over post-synaptic neurons (output neurons of layer 2)
            for i in range(net.output_size):
                # The post synaptic neuron j has output phi_i
                r_post = phi[:, i] * a2[:, i]  # r_post.shape is [batch_size]

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

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    task_losses.append(avg_epoch_loss)
    if epoch % 1 == 0:
        pbar.set_postfix(avg_epoch_loss=avg_epoch_loss)
all_losses.extend(task_losses)
