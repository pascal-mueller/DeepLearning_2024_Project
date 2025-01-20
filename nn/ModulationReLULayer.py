import torch
import torch.nn as nn


class ModulationReLULayer(nn.Module):
    def __init__(self, size, a=1.2):
        super().__init__()

        self.size = size
        self.a = a
        self.control_signals = torch.ones(size)

    def forward(self, x):
        foo = torch.relu(x)

        try:
            return self.control_signals * foo
        except:
            print(self.control_signals.shape, " * ", foo.shape)
            breakpoint()

    def set_control_signals(self, signals):
        self.control_signals = torch.clamp(signals, min=1.0 / self.a, max=self.a)

    def get_control_signals(self):
        return self.control_signals
