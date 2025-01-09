import torch
import torch.nn as nn


class ControlNet(nn.Module):
    def __init__(self, input_size=30, hidden_size=40, output_size=22, a=1.2):
        super(ControlNet, self).__init__()
        self.input_size = input_size  # 8 input + 20 hidden + 2 output
        self.hidden_size = hidden_size
        self.output_size = output_size  # Control signals for hidden + output neurons
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.a = a

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x + 1)
        x = torch.clamp(x, min=1.0 / self.a, max=self.a)

        return x
