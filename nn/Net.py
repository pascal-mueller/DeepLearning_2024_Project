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
        self.layer1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.h1_mrelu = ModulationReLULayer(self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.h2_mrelu = ModulationReLULayer(self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.h3_mrelu = ModulationReLULayer(self.hidden_size)
        self.layer4 = nn.Linear(self.hidden_size, self.output_size, bias=False)

    def forward(self, x):
        x = self.flatten(x)
        h1 = self.layer1(x)
        h1_activated = self.h1_mrelu(h1)
        h2 = self.layer2(h1_activated)
        h2_activated = self.h2_mrelu(h2)
        h3 = self.layer3(h2_activated)
        h3_activated = self.h3_mrelu(h3)
        out = self.layer4(h3_activated)

        # Question: Why do we apply ReLu after the output layer?
        # h2_activated = self.output_activations(h2)

        if self.softmax:
            return torch.softmax(h3_activated, dim=-1)
        else:
            return out

    def set_control_signals(self, signals):
        h1_signals = signals[:, : self.hidden_size]
        h2_signals = signals[:, self.hidden_size : 2 * self.hidden_size]
        h3_signals = signals[:, 2 * self.hidden_size :]

        self.h1_mrelu.set_control_signals(h1_signals)
        self.h2_mrelu.set_control_signals(h2_signals)
        self.h3_mrelu.set_control_signals(h3_signals)

    # def get_hidden_features(self, x):
    #     x = self.flatten(x)
    #     h1 = self.layer1(x)

    #     return self.hidden_activations(h1).detach().numpy()

    # def get_weights(self):
    #     return [
    #         self.layer1.weight.detach().numpy(),
    #         self.layer2.weight.detach().numpy(),
    #     ]

    def reset_control_signals(self):
        device = self.layer1.weight.device
        # self.hidden_activations.set_control_signals(
        #     torch.ones(self.hidden_size).to(device)
        # )
        # self.output_activations.set_control_signals(
        #     torch.ones(self.output_size).to(device)
        # )

        self.h1_mrelu.set_control_signals(torch.ones(self.hidden_size).to(device))
        self.h2_mrelu.set_control_signals(torch.ones(self.hidden_size).to(device))
        self.h3_mrelu.set_control_signals(torch.ones(self.hidden_size).to(device))
