import torch
import torch.nn as nn
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

for data, labels in train_loader:
    inner_epoch_correct = None
    inner_epoch_cvg = None

    # Get current network activities
    with torch.no_grad():
        net.reset_control_signals()
        h1 = net.layer1(net.flatten(data))
        output = net(data)
        current_activities = torch.cat([net.flatten(data), h1, output], dim=1)

    old_loss = float("inf")
    for inner_epoch in range(100):
        control_optimizer.zero_grad()
        net_optimizer.zero_grad()

        control_signals = control_net(current_activities)
        net.set_control_signals(control_signals)

        output = net(data)  # net is excluded from the graph

        from torchviz import make_dot

        dot = make_dot(output, params=dict(net.named_parameters()))
        dot.render("model_graph", format="png")

        # hardcoded label
        control_loss = criterion(output, labels)

        # l1_reg = l1_lambda * (net(data) - label).abs().sum(dim=1).mean()

        control_loss.backward()

        control_optimizer.step()
        net_optimizer.step()

        # print("\n")
        # print_info(f"Inner Epoch: {inner_epoch}")
        # # print(f"  Control Signals: {control_signals}")
        # print(f"  Output: {torch.argmax(output)}        Label: {label.item()}")
        # # print(f"  Control Loss: {control_loss.item()}")
        # print(
        #     f"  control_net.layer1.weight.grad: {control_net.layer1.weight.grad.mean()}"
        # )
        # print(
        #     f"  control_net.layer2.weight.grad: {control_net.layer2.weight.grad.mean()}"
        # )
        # # print(f"  control_net.layer1.weight: {control_net.layer1.weight}")
        # # print(f"  control_net.layer2.weight: {control_net.layer2.weight}")
        # print(f"  net.layer1.weight.grad: {net.layer1.weight.grad.mean()}")
        # print(f"  net.layer2.weight.grad: {net.layer2.weight.grad.mean()}")
        # # print(f"  net.layer1.weight: {net.layer1.weight}")
        # # print(f"  net.layer2.weight: {net.layer2.weight}")

        # print(f"  Total Control Loss: {control_loss.item()}")

        if (
            torch.argmax(output, dim=1) == labels
        ).all() and inner_epoch_correct is None:
            inner_epoch_correct = inner_epoch

        if abs(old_loss - control_loss.item()) < control_threshold:
            inner_epoch_cvg = inner_epoch
            break

        old_loss = control_loss.item()

    if inner_epoch_correct is None:
        print(
            f"Fail! Used up all inner epochs. {torch.argmax(output, dim=1).float().mean()} == {labels.float().mean()}"
        )
    else:
        print(
            f"Correct at inner epoch {inner_epoch_correct}   -->    {torch.argmax(output, dim=1).float().mean()} == {labels.float().mean()}"
        )

    if inner_epoch_cvg is None:
        print(f"Failed to converge")
    else:
        print(f"Converged at inner epoch {inner_epoch_cvg}")

    print("\n")

    inner_epoch_correct = None
    inner_epoch_cvg = None
