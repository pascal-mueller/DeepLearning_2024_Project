import torch.nn as nn
import torch.nn.functional as F


# We use the best performing MLPClassifier in https://arxiv.org/pdf/1708.07747
class FashionMNIST(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super(FashionMNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        h1 = self.fc1(x)
        h1_act = self.relu(h1)
        out = self.fc2(h1_act)

        return out
