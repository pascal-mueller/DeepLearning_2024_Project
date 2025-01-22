import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nn.Net import Net
from nn.ControlNet import ControlNet

from HelpTensor import HelpTensor


data = torch.tensor(
    [[[0.9645, -0.0271], [-0.1611, 0.9617], [0.5719, 0.5491], [0.2849, 0.8134]]],
    requires_grad=True,
)

label = torch.tensor([0.0, 1.0])


# plt.scatter(data[:, 0], data[:, 1])
# plt.show()


num_epochs = 50
inner_epochs = 156
learning_rate = 0.001
control_lr = 0.001
control_threshold = 1.103321094318002e-8
l1_lambda = 0.0


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

# Get current network activities
with torch.no_grad():
    net.reset_control_signals()
    h1 = net.layer1(net.flatten(data))
    output = net(data)
    current_activities = torch.cat([net.flatten(data), h1, output], dim=1)

control_optimizer.zero_grad()
net_optimizer.zero_grad()

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
control_loss.backward()

control_optimizer.step()
net_optimizer.step()

print("\n")
print(f"  Control Signals: {control_signals}")
print(f"  Output: {output}")
print(f"  Control Loss: {control_loss.item()}")
print(f"  control_net.layer1.weight.grad: {control_net.layer1.weight.grad}")
print(f"  control_net.layer2.weight.grad: {control_net.layer2.weight.grad}")
print(f"  control_net.layer1.weight: {control_net.layer1.weight}")
print(f"  control_net.layer2.weight: {control_net.layer2.weight}")
print(f"  net.layer1.weight.grad: {net.layer1.weight.grad}")
print(f"  net.layer2.weight.grad: {net.layer2.weight.grad}")
print(f"  net.layer1.weight: {net.layer1.weight}")
print(f"  net.layer2.weight: {net.layer2.weight}")

print(f"  Total Control Loss: {control_loss.item()}")
