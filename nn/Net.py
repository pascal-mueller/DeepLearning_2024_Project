import torch
import torch.nn as nn
from nn.ModulationReLULayer import ModulationReLULayer


class Net(nn.Module):
    def __init__(self, input_size=8, hidden_size=20, output_size=2, softmax=False):
        super().__init__()

        self.softmax = softmax
        # Question: How big can we choose the network? At some point, it might
        # just memorize instead of learn no?
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_activations = ModulationReLULayer(self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        self.output_activations = ModulationReLULayer(self.output_size)

    def forward(self, x):
        x = self.flatten(x)
        h1 = self.layer1(x)
        h1_activated = self.hidden_activations(h1)
        out = self.layer2(h1_activated)

        # Question: Why do we apply ReLu after the output layer?
        out_activated = self.output_activations(out)

        if self.softmax:
            return torch.softmax(out_activated, dim=-1)
        else:
            return out_activated

    def set_control_signals(self, signals):
        hidden_signals, output_signals = (
            signals[:, : self.hidden_size],
            signals[:, self.hidden_size :],
        )

        self.hidden_activations.set_control_signals(hidden_signals)
        self.output_activations.set_control_signals(output_signals)

    def get_hidden_features(self, x):
        x = self.flatten(x)
        h1 = self.layer1(x)

        return self.hidden_activations(h1).detach().numpy()

    def get_weights(self):
        return [
            self.layer1.weight.detach().numpy(),
            self.layer2.weight.detach().numpy(),
        ]

    def reset_control_signals(self):
        device = self.layer1.weight.device
        self.hidden_activations.set_control_signals(
            torch.ones(self.hidden_size).to(device)
        )
        self.output_activations.set_control_signals(
            torch.ones(self.output_size).to(device)
        )
