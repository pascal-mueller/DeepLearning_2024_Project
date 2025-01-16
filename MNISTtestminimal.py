import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from nn.Net import Net
from nn.ControlNet import ControlNet
from utils.colored_prints import *
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

trainset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
testset = datasets.MNIST(root="data", train=False, transform=transform, download=True)


# plt.scatter(data[:, 0], data[:, 1])
# plt.show()


num_epochs = 50
inner_epochs = 156
learning_rate = 0.001
control_lr = 0.001
control_threshold = 1e-3
l1_lambda = 0.0


# Size of all the "activities" from Net we use as input
input_size_net = 784  # Flattened image: 28 x 28
hidden_size_net = 100
output_size_net = 10
hidden_size_control = 100

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


train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
test_loader = DataLoader(testset, batch_size=512, shuffle=False)


def evaluate_model(net, control_net, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            net.reset_control_signals()
            h1 = net.layer1(net.flatten(batch_data))
            output = net(batch_data)
            current_activities = torch.cat([net.flatten(batch_data), h1, output], dim=1)

            control_signals = control_net(current_activities)
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
    l1_lambda,
):
    pbar = tqdm(range(num_epochs), desc=f"Epochs", leave=False)

    for epoch in pbar:
        print(f"Press any key to start epoch {epoch}")
        input()
        batch_losses = []

        for batch_data, batch_labels in train_loader:
            inner_epoch_correct = None
            inner_epoch_cvg = None

            # Get current network activities
            with torch.no_grad():
                net.reset_control_signals()
                h1 = net.layer1(net.flatten(batch_data))
                output = net(batch_data)
                current_activities = torch.cat(
                    [net.flatten(batch_data), h1, output], dim=1
                )

            old_loss = float("inf")
            for inner_epoch in range(100):
                control_optimizer.zero_grad()
                net_optimizer.zero_grad()  # TODO: Do I need this?

                control_signals = control_net(current_activities)
                net.set_control_signals(control_signals)

                output = net(batch_data)  # net is excluded from the graph

                # hardcoded label
                control_loss = criterion(output, batch_labels)

                # l1_reg = l1_lambda * (net(data) - label).abs().sum(dim=1).mean()

                control_loss.backward()

                control_optimizer.step()
                net_optimizer.step()

                if (
                    torch.argmax(output, dim=1) == batch_labels
                ).all() and inner_epoch_correct is None:
                    inner_epoch_correct = inner_epoch

                if abs(old_loss - control_loss.item()) < control_threshold:
                    inner_epoch_cvg = inner_epoch
                    break

                old_loss = control_loss.item()

            acc = (
                torch.sum(torch.argmax(output, dim=1) == batch_labels).item()
                / batch_labels.size(0)
                * 100
            )
            if acc < 80:
                print_error(f"Fail! {acc:.2f}% at inner_epoch {inner_epoch}")
            else:
                print_info(f"Win {acc:.2f}% at inner_epoch {inner_epoch}")

            if inner_epoch_cvg is None:
                print(f"Failed to converge")
            else:
                print(f"Converged at inner epoch {inner_epoch_cvg}")

            print("\n")

            inner_epoch_correct = None
            inner_epoch_cvg = None

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
        accuracy = evaluate_model(net, control_net, test_loader)
        print(f"Epoch {epoch}  Loss: {epoch_loss}  Accuracy: {accuracy:.2f}%")


train_model(
    net,
    control_net,
    train_loader,
    criterion,
    control_optimizer,
    net_optimizer,
    control_threshold,
    l1_lambda,
)
